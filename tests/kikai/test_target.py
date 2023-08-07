import torch
import torch.nn as nn

from zenkai import IO, ThLoss
from zenkai.kikai import target_prop


class TestStandardTargetPropLoss:

    def test_standard_target_prop_loss_calculates_loss(self):

        mse = nn.MSELoss()
        x = IO(torch.rand(4, 4), torch.rand(4, 4))
        t = IO(torch.rand(4, 4))
        loss = target_prop.StandardTargetPropLoss(ThLoss('mse'))
        assert (loss(x, t) == mse(x[1], t[0])).all()


class TestRegTargetPropLoss:

    def test_standard_target_prop_loss_calculates_loss(self):

        mse = nn.MSELoss()
        x = IO(torch.rand(4, 4), torch.rand(4, 4))
        t = IO(torch.rand(4, 4))
        loss = target_prop.RegTargetPropLoss(ThLoss('mse'), ThLoss('mse', weight=0.1))
        assert (loss(x, t) == (mse(x[1], t[0]) + 0.1 * mse(x[0], x[1]))).all()
