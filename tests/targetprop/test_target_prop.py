# TODO: Add tests for TargetProp

from zenkai import targetprob
from zenkai.kaku import _grad
import zenkai
from zenkai import IO, State, iou, LMode
from zenkai.utils import _params as params
from torch import nn
import torch


class TargetPropExample1(targetprob.TPLayerLearner):

    def __init__(self, in_features: int, out_features: int):

        forward = _grad.GradIdxLearner(
            nn.Linear(in_features, out_features), zenkai.OptimFactory('SGD', lr=1e-3).comp(), zenkai.NNLoss('MSELoss')
        )
        reverse  = _grad.GradIdxLearner(
            nn.Linear(out_features, in_features), zenkai.OptimFactory('SGD', lr=1e-3).comp(), zenkai.NNLoss('MSELoss')
        )
        super().__init__(
            forward, reverse, cat_x=False
        )
        

class TestTargetPropLearner(object):

    def test_target_prop_learner_updates(self):

        learner = TargetPropExample1(2, 4)
        state = zenkai.State()
        x = zenkai.iou(torch.randn(4, 2))
        t = zenkai.iou(torch.randn(4, 4))
        y = learner.forward_io(x, state)

        before = params.get_model_params(learner.forward_learner)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        assert (
            before !=
            params.get_model_params(learner._forward_learner)
        ).any()

    def test_target_prop_learner_updates_reverse(self):

        learner = TargetPropExample1(2, 4)
        x = iou(torch.randn(4, 2))
        t = iou(torch.randn(4, 4))

        state = State()
        y = learner.forward_io(x, state)

        before = params.get_model_params(learner.reverse_learner)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        assert (
            before !=
            params.get_model_params(learner.reverse_learner)
        ).any()

    def test_target_prop_learner_does_not_update_reverse_when_turned_off(self):

        learner = TargetPropExample1(2, 4)
        x = iou(torch.randn(4, 2))
        t = iou(torch.randn(4, 4))
        state = State()
        y = learner.forward_io(x, state)
        learner.reverse_update = False
        before = params.get_model_params(learner.reverse_learner)
        learner.reverse_update
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        assert (
            before ==
            params.get_model_params(learner.reverse_learner)
        ).all()
    
    def test_target_prop_learner_does_not_update_forward_when_turned_off(self):

        learner = TargetPropExample1(2, 4)
        x = iou(torch.randn(4, 2))
        t = iou(torch.randn(4, 4))
        state = State()
        y = learner.forward_io(x, state)
        learner.forward_update = False

        before = params.get_model_params(learner.forward_learner)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        assert (
            before ==
            params.get_model_params(learner.forward_learner)
        ).all()
    
    def test_target_prop_learner_returns_valid_x(self):

        learner = TargetPropExample1(2, 4)
        state = State()
        x = iou(torch.randn(4, 2))
        t = iou(torch.randn(4, 4))
        y = learner.forward_io(x, state)

        learner.forward_update = False
        learner.accumulate(x, t, state)
        x_prime = learner.step_x(x, t, state)
        assert (
            x.f != x_prime.f
        ).any()

    def test_chained_target_prop_updates_forward(self):

        learner = TargetPropExample1(2, 4)
        learner2 = TargetPropExample1(4, 4)

        learner.lmode_(LMode.WithStep)
        learner2.lmode_(LMode.WithStep)

        before = params.get_model_params(learner.forward_learner)
        x = torch.randn(4, 2)
        t = torch.randn(4, 4)
        y = learner(x)
        y = learner2(y)

        (y - t).pow(2).sum().backward()

        assert (
            before !=
            params.get_model_params(learner.forward_learner)
        ).any()

    def test_chained_target_prop_updates_reverse(self):

        learner = TargetPropExample1(2, 4)
        learner2 = TargetPropExample1(4, 4)

        learner.lmode_(LMode.WithStep)
        learner2.lmode_(LMode.WithStep)

        before = params.get_model_params(learner.reverse_learner)
        before2 = params.get_model_params(learner2.reverse_learner)
        x = torch.randn(4, 2)
        t = torch.randn(4, 4)
        y = learner(x)
        y = learner2(y)

        (y - t).pow(2).sum().backward()

        assert (
            before !=
            params.get_model_params(learner.reverse_learner)
        ).any()
        assert (
            before2 !=
            params.get_model_params(learner2.reverse_learner)
        ).any()


