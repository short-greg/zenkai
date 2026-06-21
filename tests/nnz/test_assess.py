import pytest
import torch
import torch.nn as nn

from zenkai._core import iou
from zenkai.nnz import _assess as _evaluation
from zenkai.nnz._assess import MulticlassLoss, NNLoss


class TestThLoss:

    def test_th_loss_outputs_correct_loss_with_mse_and_no_reduction(self):

        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        loss = NNLoss("MSELoss", "none")
        evaluation = loss(iou(x), iou(t))
        assert (evaluation == nn.MSELoss(reduction="none")(x, t)).all()

    def test_th_loss_outputs_correct_loss_with_mse_and_mean_reduction(self):

        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        loss = NNLoss("MSELoss", "mean")
        evaluation = loss(iou(x), iou(t))
        assert (evaluation == nn.MSELoss(reduction="mean")(x, t)).all()

    def test_th_loss_outputs_correct_loss_with_mseloss_and_mean_reduction(self):

        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        loss = NNLoss("MSELoss", "mean")
        evaluation = loss(iou(x), iou(t))
        assert (evaluation == nn.MSELoss(reduction="mean")(x, t)).all()

    def test_th_loss_fails_with_invalid_reduction(self):

        with pytest.raises(KeyError):
            NNLoss("XLoss", "mean")

    def test_th_loss_outputs_correct_loss_with_mse_and_mean_override_reduction(self):

        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        loss = NNLoss("MSELoss", "none")
        evaluation = loss(iou(x), iou(t), "mean")
        assert (evaluation == nn.MSELoss(reduction="mean")(x, t)).all()

    def test_assess_returns_assessment(self):

        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        loss = NNLoss("MSELoss", "none")
        evaluation = loss.assess(iou(x), iou(t), "mean")
        assert isinstance(evaluation, torch.Tensor)

    def test_maximize_returns_true_if_maximize(self):

        loss = NNLoss("MSELoss", "mean", maximize=True)
        assert loss.maximize is True


class TestLookup:

    def test_lookup_gets_mse_loss(self):

        mse_loss = _evaluation.lookup_loss("MSELoss")
        assert mse_loss == nn.MSELoss

    def test_lookup_returns_error_if_invalid(self):

        with pytest.raises(KeyError):
            _evaluation.lookup_loss("XLoss")


class TestMulticlassLoss:

    def test_forward_outputs_classification_rate(self):

        y = torch.randint(0, 8, (8,)).float()
        t = torch.randint(0, 8, (8,)).float()
        criterion = MulticlassLoss()
        loss = criterion(y, t)
        assert loss.item() == (y == t).float().mean().item()

    def test_backward_returns_difference(self):

        y = torch.randint(0, 8, (8,)).float()
        y.requires_grad_()
        t = torch.randint(0, 8, (8,)).float()
        criterion = MulticlassLoss()
        loss = criterion(y, t)
        loss.backward()
        assert (y.grad == (y - t)).all()
