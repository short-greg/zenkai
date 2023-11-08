import torch

from zenkai import utils
from zenkai.kaku import IO, State
from zenkai.kikai._null import NullLearner


class TestNullLearner:
    
    def test_step_does_not_update_parameters(self):

        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        learner = NullLearner()
        state = State()

        before = utils.get_model_parameters(learner)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        after = utils.get_model_parameters(learner)
        assert (before == after) and before is None
    
    def test_step_x_does_not_change_y(self):

        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        learner = NullLearner()
        state = State()

        learner.accumulate(x, t, state)
        x_prime = learner.step_x(x, t, state)
        assert (x_prime.f == x.f).all()

    def test_forward_outputs_x(self):

        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        learner = NullLearner()
        state = State()

        y = learner(x, state)
        assert (y.f == x.f).all()
