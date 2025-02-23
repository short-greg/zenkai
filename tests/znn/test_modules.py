import torch
from zenkai.nnz import FreezeDropout, Null, Updater
from zenkai.thz import update_momentum


class TestFreezeDropout:
    def test_freeze_dropout_outputs_same_value(self):

        torch.manual_seed(1)
        dropout = FreezeDropout(0.1, True)
        x = torch.rand(2, 2)
        y = dropout(x)
        y2 = dropout(x)
        assert (y == y2).all()

    def test_freeze_dropout_outputs_same_value_when_testing(self):

        torch.manual_seed(1)
        dropout = FreezeDropout(0.1, True)
        dropout.eval()
        x = torch.rand(2, 2)
        y = dropout(x)
        y2 = dropout(x)
        assert (y == y2).all()

    def test_freeze_dropout_outputs_different_values_with_unfrozen(self):

        torch.manual_seed(1)
        dropout = FreezeDropout(0.1, False)
        x = torch.rand(2, 2)
        y2 = dropout(x)
        y = dropout(x)
        assert (y != y2).any()

    def test_freeze_dropout_outputs_different_value_after_unfreezing(self):

        torch.manual_seed(1)
        dropout = FreezeDropout(0.1, True)
        x = torch.rand(2, 2)
        y = dropout(x)
        dropout.freeze = False
        y2 = dropout(x)
        assert (y != y2).any()


class TestNull:
    
    def test_null_forward(self):

        null = Null()
        x = torch.randn(2, 2)
        y = null(x)
        assert x is y

    def test_null_forward_with_multi(self):

        null = Null()
        x = torch.randn(2, 2)
        x1 = torch.randn(2, 2)
        y, y1 = null(x, x1)
        assert x is y
        assert x1 is y1

    def test_null_reverse(self):

        null = Null()
        x = torch.randn(2, 2)
        y = null.reverse(x)
        assert x is y

    def test_null_reverse_with_multi(self):

        null = Null()
        x = torch.randn(2, 2)
        x1 = torch.randn(2, 2)
        y, y1 = null.reverse(x, x1)
        assert x is y
        assert x1 is y1


class TestUpdater:

    def test_updater_returns_x(self):

        updater = Updater(
            update_momentum, momentum=0.9
        )
        x = torch.randn(4, 4)
        y1 = updater(x)
        assert (y1 == x).all()

    def test_cur_val_is_x_if_first_val(self):

        updater = Updater(
            update_momentum, momentum=0.9
        )
        x = torch.randn(4, 4)
        updater(x)
        assert (updater.cur_val is x)

    def test_cur_val_is_updated_after_one(self):

        updater = Updater(
            update_momentum, momentum=0.9
        )
        x = torch.randn(4, 4)
        x2 = torch.randn(4, 4)
        updater(x)
        updater(x2)
        assert (updater.cur_val != x).any()
