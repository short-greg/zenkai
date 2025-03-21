import torch
from zenkai.lm._io2 import IO as IO
from .test_grad import THGradLearnerT1
from zenkai.lm._state import State


import pytest
import torch
from zenkai.lm._ensemble import EnsembleVoterLearner
from zenkai.lm._io2 import IO
from zenkai.lm._state import State
from zenkai import utils


@pytest.fixture
def ensemble():
    return EnsembleVoterLearner(
        lambda: THGradLearnerT1(3, 4), n_keep=3
    )


class TestEnsemble:

    def test_forward_nn(self, ensemble):
        x = IO([torch.randn(8, 3)])
        state = State()
        result = ensemble.forward_nn(x, state)
        assert isinstance(result, torch.Tensor)
        assert len(result) == 1 or len(result) == 3
        assert all(isinstance(r, torch.Tensor) for r in result)

    def test_forward_io(self, ensemble):
        x = IO([torch.randn(8, 3)])
        state = State()
        result = ensemble.forward_io(x, state)
        print(result)
        assert isinstance(result, IO)
        assert len(result) == 1 or len(result) == 3
        assert all(isinstance(r, torch.Tensor) for r in result)

    def test_accumulate(self, ensemble):
        x = IO([torch.randn(8, 3)])
        t = IO([torch.randn(8, 4)])
        state = State()
        ensemble.forward_io(x, state)
        ensemble.accumulate(x, t, state)
        for learner in ensemble.learners:
            grads = utils.to_gradvec(learner)
            assert (grads != 0.0).any()

    def test_step(self, ensemble):
        x = IO([torch.randn(8, 3)])
        t = IO([torch.randn(8, 4)])
        state = State()
        ensemble.forward_io(x, state)
        ensemble.accumulate(x, t, state)
        befores = []
        for learner in ensemble.learners:
            befores.append(utils.to_pvec(learner))
        ensemble.step(x, t, state)
        for before, learner in enumerate(ensemble.learners):
            assert (utils.to_pvec(learner) != before).any()

    def test_step_x(self, ensemble):
        x = IO([torch.randn(8, 3)])
        t = IO([torch.randn(8, 4)])
        state = State()
        ensemble.forward_io(x, state)
        ensemble.accumulate(x, t, state)
        result = ensemble.step_x(x, t, state)
        assert isinstance(result, IO)
