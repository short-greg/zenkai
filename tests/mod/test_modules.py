import torch

from zenkai.mod import _classify


class TestArgmax:

    def test_argmax_returns_last(self):

        argmax = _classify.Argmax()
        x = torch.cumsum(torch.rand(2, 4), dim=-1)
        result = argmax(x)
        assert result[0] == 3

    def test_argmax_returns_first(self):

        argmax = _classify.Argmax()
        x = torch.cumsum(torch.rand(2, 4), dim=-1).sort(-1, True)[0]
        result = argmax(x)
        assert result[0] == 0


class TestSign:

    def test_sign_outputs_ones_and_neg_ones(self):

        x = torch.randn(2, 4, requires_grad=True)
        x.retain_grad()
        sign = _classify.Sign(False)
        y = sign(x)
        assert ((y == -1) | (y == 1)).all()

    def test_sign_outputs_ones_and_neg_ones_with_grad(self):

        x = torch.randn(2, 4, requires_grad=True)
        x.retain_grad()
        sign = _classify.Sign(True)
        y = sign(x)
        assert ((y == -1) | (y == 1)).all()

    def test_sign_with_no_grad(self):

        x = torch.randn(2, 4, requires_grad=True)
        x.retain_grad()
        sign = _classify.Sign(False)
        y = sign(x)
        y.sum().backward()
        assert (x.grad == 0).all()

    def test_sign_with_grad(self):

        x = torch.randn(2, 4, requires_grad=True)
        x.retain_grad()
        sign = _classify.Sign(True)
        y = sign(x)
        y.sum().backward()
        assert (x.grad != 0).any()


class TestBinary:

    def test_binary_outputs_ones_and_zeros(self):

        x = torch.randn(2, 4, requires_grad=True)
        x.retain_grad()
        sign = _classify.Binary(False)
        y = sign(x)
        assert ((y == 0) | (y == 1)).all()

    def test_binary_outputs_ones_and_zeros_with_grd(self):

        x = torch.randn(2, 4, requires_grad=True)
        x.retain_grad()
        sign = _classify.Binary(True)
        y = sign(x)
        assert ((y == 0) | (y == 1)).all()

    def test_binary_grad_with_no_grad(self):

        x = torch.randn(2, 4, requires_grad=True)
        x.retain_grad()
        sign = _classify.Binary(False)
        y = sign(x)
        y.sum().backward()
        assert (x.grad == 0).all()

    def test_binary_grad_with_grad(self):

        x = torch.randn(2, 4, requires_grad=True)
        x.retain_grad()
        sign = _classify.Binary(True)
        y = sign(x)
        y.sum().backward()
        assert (x.grad != 0).any()


class TestFreezeDropout:

    def test_freeze_dropout_outputs_same_value(self):

        torch.manual_seed(1)
        dropout = _classify.FreezeDropout(0.1, True)
        x = torch.rand(2, 2)
        y = dropout(x)
        y2 = dropout(x)
        assert (y == y2).all()

    def test_freeze_dropout_outputs_same_value_when_testing(self):

        torch.manual_seed(1)
        dropout = _classify.FreezeDropout(0.1, True)
        dropout.eval()
        x = torch.rand(2, 2)
        y = dropout(x)
        y2 = dropout(x)
        assert (y == y2).all()

    def test_freeze_dropout_outputs_different_values_with_unfrozen(self):

        torch.manual_seed(1)
        dropout = _classify.FreezeDropout(0.1, False)
        x = torch.rand(2, 2)
        y2 = dropout(x)
        y = dropout(x)
        assert (y != y2).any()

    def test_freeze_dropout_outputs_different_value_after_unfreezing(self):

        torch.manual_seed(1)
        dropout = _classify.FreezeDropout(0.1, True)
        x = torch.rand(2, 2)
        y = dropout(x)
        dropout.freeze = False
        y2 = dropout(x)
        assert (y != y2).any()
