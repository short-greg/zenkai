import torch

from zenkai.utils import _params as utils
from zenkai.kaku._io2 import IO2 as IO, iou
from zenkai.kaku._state import Meta
from zenkai.kaku._null import NullLearner


class TestNullLearner:

    def test_step_does_not_update_parameters(self):

        x = iou(torch.rand(2, 2))
        t = iou(torch.rand(2, 3))
        learner = NullLearner()

        state = Meta()
        before = utils.get_model_params(learner)
        learner.forward_io(x, state)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        after = utils.get_model_params(learner)
        assert (before == after) and before is None

    def test_step_x_does_not_change_y(self):

        x = iou(torch.rand(2, 2))
        t = iou(torch.rand(2, 3))
        learner = NullLearner()

        state = Meta()
        learner.forward_io(x, state)
        learner.accumulate(x, t, state)
        x_prime = learner.step_x(x, t, state)
        assert (x_prime.f == x.f).all()

    def test_forward_outputs_x(self):

        x = iou(torch.rand(2, 2))
        learner = NullLearner()

        state = Meta()
        y = learner.forward_io(x, state)
        assert (y.f == x.f).all()
