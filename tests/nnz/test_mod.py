import torch

from zenkai._core import update_momentum
from zenkai.nnz._mod import Updater


class TestUpdater:
    def test_updater_returns_x(self):

        updater = Updater(update_momentum, momentum=0.9)
        x = torch.randn(4, 4)
        y1 = updater(x)
        assert (y1 == x).all()

    def test_cur_val_is_x_if_first_val(self):

        updater = Updater(update_momentum, momentum=0.9)
        x = torch.randn(4, 4)
        updater(x)
        assert updater.cur_val is x

    def test_cur_val_is_updated_after_one(self):

        updater = Updater(update_momentum, momentum=0.9)
        x = torch.randn(4, 4)
        x2 = torch.randn(4, 4)
        updater(x)
        updater(x2)
        assert (updater.cur_val != x).any()
