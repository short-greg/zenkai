import pytest
import torch
import torch.nn as nn

from zenkai.kaku import _assess as _evaluation, ThLoss, IO


class TestAssessment:

    def test_mean_computes_correct_mean_for_dim_of_None(self):
        t = torch.rand([4])
        assessment = _evaluation.Assessment(t)
        assert assessment.mean().item() == t.mean().item()

    def test_mean_computes_correct_mean_for_dim_0(self):
        t = torch.rand([4, 3])
        assessment = _evaluation.Assessment(t)
        assert (assessment.mean(dim=0).value == t.mean(dim=0)).all()

    def test_mean_computes_correct_mean_for_dim_of_None(self):
        t = torch.rand([4])
        assessment = _evaluation.Assessment(t)
        assert assessment.sum().item() == t.sum().item()

    def test_sum_computes_correct_mean_for_dim_0(self):
        t = torch.rand([4, 3])
        assessment = _evaluation.Assessment(t)
        assert (assessment.sum(dim=0).value == t.sum(dim=0)).all()

    def test_detach_removes_the_grad_requirement(self):

        x = torch.rand([4, 3], requires_grad=True)
        t = x * 2
        assessment = _evaluation.Assessment(t)
        assert assessment.detach().value.requires_grad is False

    def test_numpy_returns_an_ndarray(self):

        x = torch.rand([4, 3])
        assessment = _evaluation.Assessment(x)
        assert (assessment.numpy() == x.numpy()).all()

    def test_item_returns_the_value_in_scalar_tensor(self):

        x = torch.tensor(2)
        assessment = _evaluation.Assessment(x)
        assert (assessment.item() == 2)

    def test_item_raises_error_if_not_dim_of_0(self):

        x = torch.tensor([2, 1])
        assessment = _evaluation.Assessment(x)
        with pytest.raises(ValueError):
            assessment.item()

    def test_backward_computes_the_grad_of_the_assessment(self):

        x = torch.tensor(2.0, requires_grad=True)
        t = x * 2
        _evaluation.Assessment(t).mean().backward()
        assert x.grad is not None

    def test_backward_raises_error_if_not_dim_of_0(self):

        x = torch.tensor([2, 1])
        assessment = _evaluation.Assessment(x)
        with pytest.raises(RuntimeError):
            assessment.backward()


class TestAssessmentDict:
    
    def test_getitem_retrieves_all_items(self):
        assessment = _evaluation.Assessment(torch.rand(2))
        assessment_dict = _evaluation.AssessmentDict(
            t=assessment
        )
        assert assessment_dict['t'] == assessment
    
    def test_items_retrieves_all_items(self):
        assessment = _evaluation.Assessment(torch.rand(2))
        d_ = dict(_evaluation.AssessmentDict(
            t=assessment
        ).items())
        assert d_['t'] == assessment

    def test_values_retrieves_all_values(self):
        assessment = _evaluation.Assessment(torch.rand(2))
        d_ = list(_evaluation.AssessmentDict(
            t=assessment
        ).values())
        assert d_[0] == assessment

    def test_mean_computes_all_means(self):
        assessment = _evaluation.Assessment(torch.rand(2))
        assessment2 = _evaluation.Assessment(torch.rand(2))
        result = _evaluation.AssessmentDict(
            t=assessment, t2=assessment2
        ).mean().sub(['t', 't2'])
        assert result['t'].value == assessment.value.mean()
        assert result['t2'].value == assessment2.value.mean()

    def test_values_retrieves_all_values(self):
        assessment = _evaluation.Assessment(torch.rand(2))
        d_ = list(_evaluation.AssessmentDict(
            t=assessment
        ).values())
        assert d_[0] == assessment


class TestThLoss:

    def test_th_loss_outputs_correct_loss_with_mse_and_no_reduction(self):

        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        loss = ThLoss("mse", 'none')
        evaluation = loss(IO(x), IO(t))
        assert (evaluation == nn.MSELoss(reduction='none')(x, t)).all()

    def test_th_loss_outputs_correct_loss_with_mse_and_mean_reduction(self):

        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        loss = ThLoss("mse", 'mean')
        evaluation = loss(IO(x), IO(t))
        assert (evaluation == nn.MSELoss(reduction='mean')(x, t)).all()
    
    def test_th_loss_outputs_correct_loss_with_mse_and_mean_override_reduction(self):

        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        loss = ThLoss("mse", 'none')
        evaluation = loss(IO(x), IO(t), 'mean')
        assert (evaluation == nn.MSELoss(reduction='mean')(x, t)).all()
