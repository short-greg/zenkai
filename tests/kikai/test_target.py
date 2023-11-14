import torch
import torch.nn as nn
from zenkai import IO, ThLoss
from zenkai.kikai import _target_prop


class TestStandardTargetPropLoss:
    def test_standard_target_prop_loss_calculates_loss(self):

        mse = nn.MSELoss()
        x = IO(torch.rand(4, 4), torch.rand(4, 4))
        t = IO(torch.rand(4, 4))
        loss = _target_prop.StandardTargetPropObjective(ThLoss("MSELoss"))
        assert (loss(x, t) == mse(x.u[1], t.f)).all()


class TestRegTargetPropLoss:
    def test_standard_target_prop_loss_calculates_loss(self):

        mse = nn.MSELoss()
        x = IO(torch.rand(4, 4), torch.rand(4, 4))
        t = IO(torch.rand(4, 4))
        loss = _target_prop.RegTargetPropObjective(
            ThLoss("MSELoss"), ThLoss("MSELoss", weight=0.1)
        )
        assert (loss(x, t) == (mse(x.u[1], t.f) + 0.1 * mse(x.f, x.u[1]))).all()
