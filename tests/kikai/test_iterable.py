import torch

from zenkai import utils
from zenkai.kaku import IO, State
from zenkai.kikai.iterable import IterStepTheta, IterStepX, IterHiddenStepTheta
from ..kaku.test_machine import SimpleLearner


class TestIterStepTheta:
    
    def test_iter_hiddenstep_updates_the_parameters_with_one_iteration(self):

        torch.manual_seed(3)
        learner1 = SimpleLearner(2, 3)
        learner2 = SimpleLearner(3, 3)    
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        iter_step = IterHiddenStepTheta(learner1, learner1, learner2, 1, 1, 1)
        state = State()
        y1 = learner1(x, state)
        learner2(y1, state)
        learner2.step(y1, t, state)

        before = utils.get_model_parameters(learner1)
        iter_step.step(x, y1, state, t)
        after = utils.get_model_parameters(learner1)
        assert (before != after).any()


class TestIterStepX:
    
    def test_iter_ste_x_updates_x_with_one_iteration(self):

        torch.manual_seed(3)
        learner1 = SimpleLearner(2, 3)
        learner2 = SimpleLearner(3, 3)    
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        iter_step = IterStepX(learner2, 1, 128)
        state = State()
        y1 = learner1(x, state)
        learner2(y1, state)
        learner2.step(y1, t, state)
        before = torch.clone(y1[0])
        x = iter_step.step_x(y1, t, state)

        assert (before != x[0]).any()


class TestIterStepHidden:

    def test_iter_outstep_updates_the_parameters_with_one_iteration(self):

        torch.manual_seed(1)
        learner1 = SimpleLearner(2, 3)  
        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        iter_step = IterStepTheta(learner1, 1, 128)
        before = utils.get_model_parameters(learner1)
        state = State()
        learner1(x, state)
        iter_step.step(x, t, state)
        after = utils.get_model_parameters(learner1)
        assert (before != after).any()
