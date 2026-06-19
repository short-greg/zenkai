import torch

from zenkai.lm._io2 import IO, iou
from zenkai.lm._lm2 import LearningMachine, LMode, State
from zenkai.lm._global_step import GlobalTargetLearner
from zenkai.utils import to_pvec
import torch.nn as nn

from .test_grad import THGradLearnerT1, THGradLearnerT2


class DummyGlobalTargetLearner(GlobalTargetLearner):

    def __init__(self, lmode: LMode=LMode.Standard):
        super().__init__(lmode)

        self.learner1 = THGradLearnerT1(2, 4)
        self.learner2 = THGradLearnerT1(4, 2)

    def forward_iter(self, x, state, **kwargs):
        
        y1 = self.learner1.forward_io(x, state.sub('sub1'))
        y2 = self.learner2.forward_io(y1, state.sub('sub2'))

        yield self.learner1, y1, state.sub('sub1')
        yield self.learner2, y2, state.sub('sub2')
    
    def optim_x(self, x, t, state):
        
        return iou(torch.randn_like(x.f))


class TestGlobalTargetLearner:

    def test_forward_outputs_correct_value(self):

        learner = DummyGlobalTargetLearner()
        x = torch.randn(4, 2)
        t = learner.learner2(learner.learner1(x))
        y = learner(x)

        assert (y == t).all()

    def test_parameters_are_all_updated_on_backward(self):

        learner = DummyGlobalTargetLearner(LMode.WithStep)
        x = torch.randn(4, 2)
        t = torch.rand(4, 2)
        y = learner(x)
        before = to_pvec(learner)
        (y - t).pow(2).mean().backward()
        after = to_pvec(learner)
    
        assert (before != after).any()

    def test_step_x_returns_new_x(self):

        learner = DummyGlobalTargetLearner(LMode.WithStep)
        x = torch.randn(4, 3)
        t = torch.rand(4, 2)
        base = nn.Linear(3, 2)
        y = learner(base(x))
        (y - t).pow(2).mean().backward()
    
        assert base.weight.grad is not None
