import torch

from zenkai.lm._io2 import IO, iou
from zenkai.lm._lm2 import LearningMachine, LMode, State
from zenkai.lm._autoencoder import AutoencodedLearner
from zenkai.utils import to_pvec
import torch.nn as nn

from .test_grad import THGradLearnerT1, THGradLearnerT2

# TODO: Currently i don't have a good way to just reconstruct
#   unless I can output the reconstruction

# TODO: Think more about how to handle this

# 1) 

class TestAutoencoderLearner:

    def test_forward_outputs_correct_value(self):

        learner = THGradLearnerT1(3, 4)
        learner2 = THGradLearnerT1(4, 3)
        autoencoded = AutoencodedLearner(
            learner, learner2, 1.0
        )
        x = torch.randn(4, 3)
        t = learner(x)
        y = autoencoded(x)
        
        assert (y == t).all()

    def test_parameters_are_all_updated_on_backward(self):

        learner = THGradLearnerT1(3, 4)
        learner2 = THGradLearnerT1(4, 3)
        autoencoded = AutoencodedLearner(
            learner, learner2, 1.0, lmode=LMode.WithStep
        )
        x = torch.randn(4, 3)
        y = autoencoded(x)
        t = torch.randn(4, 4)
        before = to_pvec(autoencoded)
        (y - t).pow(2).mean().backward()
        after = to_pvec(autoencoded)
    
        assert (before != after).any()

    def test_parameters_are_all_updated_on_backward_with_rec_weightNone(self):

        learner = THGradLearnerT1(3, 4)
        learner2 = THGradLearnerT1(4, 3)
        autoencoded = AutoencodedLearner(
            learner, learner2, None, lmode=LMode.WithStep
        )
        x = torch.randn(4, 3)
        y = autoencoded(x)
        t = torch.randn(4, 4)
        before = to_pvec(autoencoded)
        (y - t).pow(2).mean().backward()
        after = to_pvec(autoencoded)
    
        assert (before != after).any()


    def test_step_x_returns_new_x(self):


        learner = THGradLearnerT1(2, 4)
        learner2 = THGradLearnerT1(4, 2)
        autoencoded = AutoencodedLearner(
            learner, learner2, 1.0, lmode=LMode.WithStep
        )
        x = torch.randn(4, 3)
        t = torch.randn(4, 4)

        base = nn.Linear(3, 2)
        y = autoencoded(base(x))
        (y - t).pow(2).mean().backward()
    
        assert base.weight.grad is not None
