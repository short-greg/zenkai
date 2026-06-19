import torch

from zenkai._core import State
from zenkai._core import _params as utils
from zenkai._core import iou
from zenkai.lm._null import NullLearner


def _params_or_none(model):
    """``params_get`` raises on a parameter-less module, so treat that as ``None``."""
    if next(model.parameters(), None) is None:
        return None
    return utils.params_get(model)


class TestNullLearner:

    def test_step_does_not_update_parameters(self):

        x = iou(torch.rand(2, 2))
        t = iou(torch.rand(2, 3))
        learner = NullLearner()

        state = State()
        before = _params_or_none(learner)
        learner.forward_io(x, state)
        learner.accumulate(x, t, state)
        learner.step(x, t, state)
        after = _params_or_none(learner)
        assert (before == after) and before is None

    def test_step_x_does_not_change_y(self):

        x = iou(torch.rand(2, 2))
        t = iou(torch.rand(2, 3))
        learner = NullLearner()

        state = State()
        learner.forward_io(x, state)
        learner.accumulate(x, t, state)
        x_prime = learner.step_x(x, t, state)
        assert (x_prime.f == x.f).all()

    def test_forward_outputs_x(self):

        x = iou(torch.rand(2, 2))
        learner = NullLearner()

        state = State()
        y = learner.forward_io(x, state)
        assert (y.f == x.f).all()
