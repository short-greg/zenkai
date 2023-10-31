import pytest
import torch
import torch.nn as nn


from zenkai.kaku import (
    _assess as _evaluation, 
    ThLoss, IO, 
    Reduction, reduce_assessment
)


class TestReduction:

    def test_is_torch_checks_if_it_is_torch(self):
        assert Reduction.is_torch('mean')

    def test_reduction_with_mean_calculates_the_mean(self):

        x = torch.rand(4)
        reduction = Reduction.mean.reduce(x)
        assert x.mean() == reduction

    def test_reduction_with_sum_calculates_the_mean(self):

        x = torch.rand(4)
        reduction = Reduction.sum.reduce(x)
        assert x.sum() == reduction

    def test_reduction_batchmean_returns_batch_mean(self):

        x = torch.rand(4)
        reduction = Reduction.sum.reduce(x)
        assert x.sum() == reduction

    def test_reduction_batchmean_returns_batchmean(self):

        x = torch.rand(4)
        reduction = Reduction.batchmean.reduce(x)
        assert x.sum() / len(x) == reduction

    def test_reduction_batchmean_returns_samplemeans(self):

        x = torch.rand(2, 4)
        reduction = Reduction.samplemeans.reduce(x)
        assert (x.mean(dim=1) == reduction).all()



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

    def test_add_adds_two_assessments(self):
        t1 = torch.rand([4, 3])
        t2 = torch.rand([4, 3])
        assessment1 = _evaluation.Assessment(t1)
        assessment2 = _evaluation.Assessment(t2)
        assert ((assessment1 + assessment2).value == t1 + t2).all()

    def test_batch_mean_calculates(self):
        t1 = torch.rand([4, 3])
        t2 = torch.rand([4, 3])
        assessment1 = _evaluation.Assessment(t1)
        assessment2 = _evaluation.Assessment(t2)
        assert ((assessment1 + assessment2).value == t1 + t2).all()

    def test_view_changes_view_of_assessment(self):
        t1 = torch.rand([4, 3])
        assessment1 = _evaluation.Assessment(t1)
        assert ((assessment1.view(-1)).value == t1.view(-1)).all()

    def test_best_gets_max_value(self):
        t1 = torch.rand([4, 3]).cumsum(dim=1)
        assessment1 = _evaluation.Assessment(t1, maximize=True)
        best, ind = assessment1.best()
        val, ind = t1.max(dim=0)
        assert (val == best).all()

    def test_best_gets_min_value(self):
        t1 = torch.rand([4, 3]).cumsum(dim=1)
        assessment1 = _evaluation.Assessment(t1, maximize=False)
        best, ind = assessment1.best()
        val, ind = t1.min(dim=0)
        assert (val == best).all()

    def test_stack_stacks_all_tensors(self):
        t1 = _evaluation.Assessment(torch.rand([4, 3]).cumsum(dim=1), maximize=False)
        t2 = _evaluation.Assessment(torch.rand([4, 3]).cumsum(dim=1), maximize=False)
        assessment1 = _evaluation.Assessment.stack([t1, t2])
        assert len(assessment1) == 2

    def test_stack_raises_error_if_not_all_same_direction(self):
        t1 = _evaluation.Assessment(torch.rand([4, 3]).cumsum(dim=1), maximize=False)
        t2 = _evaluation.Assessment(torch.rand([4, 3]).cumsum(dim=1), maximize=True)
        with pytest.raises(ValueError):
            _evaluation.Assessment.stack([t1, t2])

    def test_to_image(self):
        t1 = _evaluation.Assessment(torch.rand([4, 3, 4]).cumsum(dim=2), maximize=False)
        t1_2d = t1.reduce_image(reduction='mean')
        assert (t1_2d.value == t1.view(4, 12).mean(1).value).all()

    def test_to_image_raises_error_if_invalid(self):
        t1 = _evaluation.Assessment(torch.rand([4, 3, 4]).cumsum(dim=2), maximize=False)
        with pytest.raises(ValueError):
            t1.reduce_image(divide_start=0, reduction='mean')
        
    def test_to_2d(self):
        t1 = _evaluation.Assessment(torch.rand([4, 3, 4]).cumsum(dim=2), maximize=False)
        t1_2d, _, _ = t1.to_2d()
        assert t1_2d.size(1) == 12


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
        loss = ThLoss("MSELoss", 'none')
        evaluation = loss(IO(x), IO(t))
        assert (evaluation == nn.MSELoss(reduction='none')(x, t)).all()

    def test_th_loss_outputs_correct_loss_with_mse_and_mean_reduction(self):

        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        loss = ThLoss("MSELoss", 'mean')
        evaluation = loss(IO(x), IO(t))
        assert (evaluation == nn.MSELoss(reduction='mean')(x, t)).all()
    
    def test_th_loss_outputs_correct_loss_with_mseloss_and_mean_reduction(self):

        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        loss = ThLoss("MSELoss", 'mean')
        evaluation = loss(IO(x), IO(t))
        assert (evaluation == nn.MSELoss(reduction='mean')(x, t)).all()
    
    def test_th_loss_fails_with_invalid_reduction(self):

        with pytest.raises(KeyError):
            ThLoss("XLoss", 'mean')

    def test_th_loss_outputs_correct_loss_with_mse_and_mean_override_reduction(self):

        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        loss = ThLoss("MSELoss", 'none')
        evaluation = loss(IO(x), IO(t), 'mean')
        assert (evaluation == nn.MSELoss(reduction='mean')(x, t)).all()

    def test_assess_returns_assessment(self):

        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        loss = ThLoss("MSELoss", 'none')
        evaluation = loss.assess(IO(x), IO(t), 'mean')
        assert isinstance(evaluation, _evaluation.Assessment)

    def test_assess_returns_assessment(self):

        x = torch.rand(4, 2)
        t = torch.rand(4, 2)
        loss = ThLoss("MSELoss", 'none')
        evaluation = loss.assess(IO(x), IO(t), 'mean')
        assert isinstance(evaluation, _evaluation.Assessment)

    def test_maximize_returns_true_if_maximize(self):

        loss = ThLoss("MSELoss", 'mean', maximize=True)
        assert loss.maximize == True


class TestLookup:

    def test_lookup_gets_mse_loss(self):

        mse_loss = _evaluation.lookup_loss('MSELoss')
        assert mse_loss == nn.MSELoss
    
    def test_lookup_returns_error_if_invalid(self):

        with pytest.raises(KeyError):
            _evaluation.lookup_loss('XLoss')
