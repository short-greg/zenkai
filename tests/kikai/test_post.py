import torch

from zenkai import IO, State
from zenkai.contain import post
from ..kaku.test_machine import SimpleLearner
from zenkai.utils import get_model_parameters


class TestStackPostStepTheta:

    def test_adv_updates_the_values(self):

        x1 = IO(torch.rand(5, 4))
        t1 = IO(torch.rand(5, 3))
        learner = SimpleLearner(4, 3)
        state = State()
        step_theta = post.StackPostStepTheta(learner)
        before = get_model_parameters(learner)
        step_theta.accumulate(x1, t1, state)
        step_theta.step(x1, t1, state)
        assert (before != get_model_parameters(learner)).any()
        
    def test_adv_updates_the_values_after_two_steps(self):

        x1 = IO(torch.rand(5, 4))
        t1 = IO(torch.rand(5, 3))
        x2 = IO(torch.rand(5, 4))
        t2 = IO(torch.rand(5, 3))
        learner = SimpleLearner(4, 3)
        state = State()
        step_theta = post.StackPostStepTheta(learner)
        before = get_model_parameters(learner)
        step_theta.accumulate(x1, t1, state)
        step_theta.accumulate(x2, t2, state)
        step_theta.step(x2, t2, state)
        assert (before != get_model_parameters(learner)).any()
        
    def test_is_sampe_after_two_steps_but_no_advances(self):

        x1 = IO(torch.rand(5, 4))
        t1 = IO(torch.rand(5, 3))
        x2 = IO(torch.rand(5, 4))
        t2 = IO(torch.rand(5, 3))
        learner = SimpleLearner(4, 3)
        state = State()
        step_theta = post.StackPostStepTheta(learner)
        before = get_model_parameters(learner)
        step_theta.accumulate(x1, t1, state)
        step_theta.accumulate(x2, t2, state)
        assert (before == get_model_parameters(learner)).all()
