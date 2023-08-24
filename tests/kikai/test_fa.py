import torch
import torch.nn as nn

from zenkai.kikai import fa
from zenkai.kaku import OptimFactory, IO, State
from zenkai.utils import get_model_parameters


class TestFA:

    def test_fa_updates_the_parameters(self):
        
        learner = fa.FALinearLearner(3, 4, optim_factory=OptimFactory('sgd', lr=1e-2), loss='mse')
        t = fa.fa_target(IO(torch.rand(3, 4)), IO(torch.rand(3, 4)))
        x = IO(torch.rand(3, 3))
        before = get_model_parameters(learner)
        learner.step(x, t, State())
        assert (get_model_parameters(learner) != before).any()

    def test_fa_backpropagates_the_target(self):
    
        learner = fa.FALinearLearner(3, 4, optim_factory=OptimFactory('sgd', lr=1e-2), loss='mse')
        t = fa.fa_target(IO(torch.rand(3, 4)), IO(torch.rand(3, 4)))
        x = IO(torch.rand(3, 3))
        x2 = learner.step_x(x, t, State())
        assert x2[0].shape == x[0].shape
        assert (x2[0] != x[0]).any()

    def test_fa_outputs_correct_value_forward(self):
        
        learner = fa.FALinearLearner(3, 4, optim_factory=OptimFactory('sgd', lr=1e-2), loss='mse')
        t = fa.fa_target(IO(torch.rand(3, 4)), IO(torch.rand(3, 4)))
        x = IO(torch.rand(3, 3))

        y = learner(x)
        assert (y[0].shape[1] == 4)


class TestBStepX:

    def test_bstepx_backpropagates_the_target(self):
    
        step_x = fa.BStepX(3, 4)
        t = fa.fa_target(IO(torch.rand(3, 4)), IO(torch.rand(3, 4)))
        x = IO(torch.rand(3, 3))
        x2 = step_x.step_x(x, t, State())
        assert x2[0].shape == x[0].shape
        assert (x2[0] != x[0]).any()
