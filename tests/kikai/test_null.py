import torch

from zenkai import utils
from zenkai.kaku import IO
from zenkai.kikai._null import NullLearner


class TestNullLearner:
    def test_step_does_not_update_parameters(self):

        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        learner = NullLearner()

        before = utils.get_model_params(learner)
        learner.accumulate(x, t)
        learner.step(x, t)
        after = utils.get_model_params(learner)
        assert (before == after) and before is None

    def test_step_x_does_not_change_y(self):

        x = IO(torch.rand(2, 2))
        t = IO(torch.rand(2, 3))
        learner = NullLearner()

        learner.accumulate(x, t)
        x_prime = learner.step_x(x, t)
        assert (x_prime.f == x.f).all()

    def test_forward_outputs_x(self):

        x = IO(torch.rand(2, 2))
        IO(torch.rand(2, 3))
        learner = NullLearner()

        y = learner(x)
        assert (y.f == x.f).all()
