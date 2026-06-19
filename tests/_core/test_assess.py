import pytest
import torch
import torch.nn as nn

from zenkai._core import _assess as _evaluation
from zenkai._core._assess import AssessmentLog, Reduction


class TestLookup:

    def test_lookup_gets_mse_loss(self):

        mse_loss = _evaluation.lookup_loss("MSELoss")
        assert mse_loss == nn.MSELoss

    def test_lookup_returns_error_if_invalid(self):

        with pytest.raises(KeyError):
            _evaluation.lookup_loss("XLoss")


class TestReduction:

    def test_is_torch_checks_if_it_is_torch(self):
        assert Reduction.is_torch("mean")

    def test_reduction_with_mean_calculates_the_mean(self):

        x = torch.rand(4)
        reduction = Reduction.mean.forward(x)
        assert x.mean() == reduction

    def test_reduction_with_sum_calculates_the_mean(self):

        x = torch.rand(4)
        reduction = Reduction.sum.forward(x)
        assert x.sum() == reduction

    def test_reduction_batchmean_returns_batch_mean(self):

        x = torch.rand(4)
        reduction = Reduction.sum.forward(x)
        assert x.sum() == reduction

    def test_reduction_batchmean_returns_batchmean(self):

        x = torch.rand(4)
        reduction = Reduction.batchmean.forward(x)
        assert x.sum() / len(x) == reduction

    def test_reduction_batchmean_returns_samplemeans(self):

        x = torch.rand(2, 4)
        reduction = Reduction.samplemeans.forward(x)
        assert (x.mean(dim=1) == reduction).all()


class TestAssessmentLog:

    def test_assessment_log_update_adds(self):
        log = AssessmentLog()
        assessment = torch.rand(1)[0]
        log.update("x", "name", "validation", assessment)
        assert log.dict["x"][None]["name"]["validation"] == assessment

    def test_assessment_log_as_assessment_dict_gets_assessment(self):
        log = AssessmentLog()
        assessment = torch.rand(1)[0]
        log.update("x", "name", "validation", assessment)
        assert log.as_assessment_dict()["name_validation"] == assessment

    def test_update_overwrites_initial_assessment(self):
        log = AssessmentLog()
        assessment = torch.rand(1)[0]
        assessment2 = torch.rand(1)[0]
        log.update("x", "name", "validation", assessment)
        log.update("x", "name", "validation", assessment2)
        assert log.as_assessment_dict()["name_validation"] == assessment2

    def test_update_overwrites_initial_assessment_even_when_keys_are_different(self):
        log = AssessmentLog()
        assessment = torch.rand(1)[0]
        assessment2 = torch.rand(1)[0]
        log.update("x", "name", "validation", assessment)
        log.update("y", "name", "validation", assessment2)
        assert log.as_assessment_dict()["name_validation"] == assessment2
