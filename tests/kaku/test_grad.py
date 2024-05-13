# 1st party

# 3rd party
import torch
from torch import nn
from zenkai import OptimFactory, ThLoss, CompOptim

# local
from zenkai.kaku._lm2 import IO2 as IO, iou, Idx2 as Idx
from zenkai.kaku import _grad
from zenkai.kaku import Meta
from zenkai.utils import _params as utils


class THGradLearnerT1(_grad.GradLearner):
    def __init__(self, in_features: int, out_features: int):
        linear = nn.Linear(in_features, out_features)
        super().__init__(
            linear,
            criterion=ThLoss(nn.MSELoss),
            optim=OptimFactory(torch.optim.Adam, lr=1e-2).comp(),
        )


class THGradLearnerT2(_grad.GradLearner):
    def __init__(self, in_features: int, out_features: int):
        linear = nn.Linear(in_features, out_features)
        super().__init__(
            linear,
            criterion=ThLoss(nn.MSELoss),
            optim=CompOptim(
                OptimFactory(torch.optim.Adam, lr=1e-2),
                OptimFactory(torch.optim.Adam, lr=1e-2)
            )
        )


class TestGradLearner1:

    def test_assess_y_uses_correct_reduction(self):

        learner = THGradLearnerT1(2, 3)
        y = IO([torch.rand(2, 3)])
        t = IO([torch.rand(2, 3)])
        result = learner.assess_y(y, t, "sum")
        target = nn.MSELoss(reduction="sum")(y.f, t.f)
        assert result.item() == target.item()

    def test_forward_does_not_detach_y(self):

        learner = THGradLearnerT1(2, 3)
        x = torch.rand(2, 2)
        y = learner.forward(x)
        assert y.grad_fn is not None

    def test_forward_io_detaches_y(self):

        learner = THGradLearnerT1(2, 3)
        x = iou(torch.rand(2, 2))
        state = Meta()
        y = learner.forward_io(x, state)
        assert y[0].grad_fn is None

    def test_step_x_updates_x(self):

        learner = THGradLearnerT1(2, 3)
        x = iou(torch.rand(2, 2))
        t = iou(torch.rand(2, 3))
        x_ = x.clone(True)
        state = Meta()
        learner.forward_io(x, state)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        x = learner.step_x(x, t, state)
        assert (x.f != x_.f).any()

    def test_step_updates_parameters(self):

        learner = THGradLearnerT1(2, 3)
        x = iou(torch.rand(2, 2))
        t = iou(torch.rand(2, 3))
        before = utils.get_model_params(learner)
        state = Meta()
        learner.forward_io(x, state)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        after = utils.get_model_params(learner)
        assert (before != after).any()


class TestTHGradLoopLearner:

    def test_assess_y_uses_correct_reduction(self):

        learner = THGradLearnerT2(2, 3)
        y = iou(torch.rand(2, 3))
        t = iou(torch.rand(2, 3))
        result = learner.assess_y(y, t, "sum")
        target = nn.MSELoss(reduction="sum")(y.f, t.f)
        assert result.item() == target.item()

    def test_step_x_updates_x(self):
        torch.manual_seed(1)
        learner = THGradLearnerT2(2, 3)
        x = iou(torch.rand(2, 2))
        og_x = x.clone()
        t = iou(torch.rand(2, 3))
        state = Meta()
        learner.forward_io(x, state)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        x = learner.step_x(x, t, state)
        assert (x.f != og_x.f).any()

    def test_step_x_updates_x_repeated(self):

        learner = THGradLearnerT2(2, 3)
        x = iou(torch.rand(4, 2))
        og_x = x.clone()
        t = iou(torch.rand(4, 3))
        idx = Idx([0, 1])
        state = Meta()
        learner.forward_io(x, state, batch_idx=idx)
        learner.accumulate(x, t, state, batch_idx=idx)
        learner.step(x, t, state, batch_idx=idx)
        x = learner.step_x(x, t, state)
        x = learner.step_x(x, t, state)
        assert (x.f != og_x.f).any()

    def test_step_updates_parameters(self):

        learner = THGradLearnerT2(2, 3)
        x = iou(torch.rand(2, 2))
        t = iou(torch.rand(2, 3))
        state = Meta()
        before = utils.get_model_params(learner)
        learner.forward_io(x, state)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        after = utils.get_model_params(learner)
        assert (before != after).any()

    def test_step_updates_parameters_repeated(self):

        learner = THGradLearnerT2(2, 3)
        x = iou(torch.rand(2, 2))
        t = iou(torch.rand(2, 3))
        before = utils.get_model_params(learner)
        state = Meta()
        learner.forward_io(x, state)
        learner.accumulate(x, t, state)
        learner.forward_io(x, state)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        after = utils.get_model_params(learner)
        assert (before != after).any()


class TestCriterionGrad:

    def test_criterion_grad_step_produces_correct_shape(self):

        state = Meta()
        learner = _grad.GradLearner(criterion=ThLoss("CrossEntropyLoss"))
        learner.step(iou(torch.rand(3, 4)), iou(torch.randint(0, 4, (3,))), state)
        assert True

    def test_criterion_grad_step_x_produces_correct_shape(self):

        learner = _grad.GradLearner(criterion=ThLoss("CrossEntropyLoss"))

        state = Meta()
        x = iou(torch.rand(3, 4))
        learner.forward_io(x, state)
        t = iou(torch.randint(0, 4, (3,)))
        learner.accumulate(x, t, state)
        x_prime = learner.step_x(x, t, state)
        assert (x.f != x_prime.f).any()
