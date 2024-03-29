# TODO: Add tests for TargetProp

from zenkai import kikai
import zenkai
from torch import nn
import torch


class TargetPropExample1(kikai.TargetPropLearner):

    def __init__(self, in_features: int, out_features: int):

        forward = kikai.GradLearner(
            nn.Linear(in_features, out_features), zenkai.OptimFactory('SGD', lr=1e-3).comp(), zenkai.ThLoss('MSELoss')
        )
        reverse  = kikai.GradLearner(
            nn.Linear(out_features, in_features), zenkai.OptimFactory('SGD', lr=1e-3).comp(), zenkai.ThLoss('MSELoss')
        )
        super().__init__(
            forward, reverse, forward, reverse, False
        )
        

class TestTargetPropLearner(object):

    def test_target_prop_learner_updates(self):

        learner = TargetPropExample1(2, 4)
        x = zenkai.IO(torch.randn(4, 2))
        t = zenkai.IO(torch.randn(4, 4))
        y = learner(x)

        before = zenkai.utils.get_model_parameters(learner.forward_module)
        learner.accumulate(x, t)
        learner.step(x, t)
        assert (
            before !=
            zenkai.utils.get_model_parameters(learner.forward_module)
        ).any()

    def test_target_prop_learner_updates_reverse(self):

        learner = TargetPropExample1(2, 4)
        x = zenkai.IO(torch.randn(4, 2))
        t = zenkai.IO(torch.randn(4, 4))
        y = learner(x)

        before = zenkai.utils.get_model_parameters(learner.reverse_module)
        learner.accumulate(x, t)
        learner.step(x, t)
        assert (
            before !=
            zenkai.utils.get_model_parameters(learner.reverse_module)
        ).any()

    def test_target_prop_learner_does_not_update_reverse_when_turned_off(self):

        learner = TargetPropExample1(2, 4)
        x = zenkai.IO(torch.randn(4, 2))
        t = zenkai.IO(torch.randn(4, 4))
        y = learner(x)

        before = zenkai.utils.get_model_parameters(learner.reverse_module)
        learner.reverse_update(False)
        learner.accumulate(x, t)
        learner.step(x, t)
        assert (
            before ==
            zenkai.utils.get_model_parameters(learner.reverse_module)
        ).all()
    
    def test_target_prop_learner_does_not_update_forward_when_turned_off(self):

        learner = TargetPropExample1(2, 4)
        x = zenkai.IO(torch.randn(4, 2))
        t = zenkai.IO(torch.randn(4, 4))
        y = learner(x)

        before = zenkai.utils.get_model_parameters(learner.forward_module)
        learner.forward_update(False)
        learner.accumulate(x, t)
        learner.step(x, t)
        assert (
            before ==
            zenkai.utils.get_model_parameters(learner.forward_module)
        ).all()
    
    def test_target_prop_learner_returns_valid_x(self):

        learner = TargetPropExample1(2, 4)
        x = zenkai.IO(torch.randn(4, 2))
        t = zenkai.IO(torch.randn(4, 4))
        y = learner(x)

        learner.forward_update(False)
        learner.accumulate(x, t)
        x_prime = learner.step_x(x, t)
        assert (
            x.f != x_prime.f
        ).any()
    