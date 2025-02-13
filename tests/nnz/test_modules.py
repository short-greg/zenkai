import torch
from zenkai.nnz import FreezeDropout


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
