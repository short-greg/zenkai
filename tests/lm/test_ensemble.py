import torch
from zenkai.lm._io2 import IO as IO
from .test_grad import THGradLearnerT1
from zenkai.lm._state import State


import pytest
import torch
from zenkai.lm._ensemble import EnsembleVoterLearner, EnsembleLearner
from zenkai.lm._io2 import IO
from zenkai.lm._state import State
from zenkai import utils


@pytest.fixture
def ensemble_voter():
    return EnsembleVoterLearner(
        lambda: THGradLearnerT1(3, 4), n_keep=3
    )


@pytest.fixture
def ensemble():
    return EnsembleLearner(
        lambda: THGradLearnerT1(3, 4), n_keep=3,
        agg=lambda io: torch.sum(io, dim=0)

    )


@pytest.fixture
def ensembleb():
    return EnsembleLearner(
        lambda: THGradLearnerT1(3, 4), n_keep=3,
        agg=lambda io: torch.sum(io, dim=0),
        use_last=False
    )


class TestEnsemble:

    def test_forward_nn(self, ensemble_voter):
        x = IO([torch.randn(8, 3)])
        state = State()
        result = ensemble_voter.forward_nn(x, state)
        assert isinstance(result, torch.Tensor)
        assert len(result) == 1 or len(result) == 3
        assert all(isinstance(r, torch.Tensor) for r in result)

    def test_forward_io(self, ensemble_voter):
        x = IO([torch.randn(8, 3)])
        state = State()
        result = ensemble_voter.forward_io(x, state)
        assert isinstance(result, IO)
        assert len(result) == 1 or len(result) == 3
        assert all(isinstance(r, torch.Tensor) for r in result)

    def test_accumulate(self, ensemble_voter):
        x = IO([torch.randn(8, 3)])
        t = IO([torch.randn(8, 4)])
        state = State()
        ensemble_voter.forward_io(x, state)
        ensemble_voter.accumulate(x, t, state)
        for learner in ensemble_voter.learners:
            grads = utils.to_gradvec(learner)
            assert (grads != 0.0).any()

    def test_step(self, ensemble_voter):
        x = IO([torch.randn(8, 3)])
        t = IO([torch.randn(8, 4)])
        state = State()
        ensemble_voter.forward_io(x, state)
        ensemble_voter.accumulate(x, t, state)
        befores = []
        for learner in ensemble_voter.learners:
            befores.append(utils.to_pvec(learner))
        ensemble_voter.step(x, t, state)
        for before, learner in enumerate(ensemble_voter.learners):
            assert (utils.to_pvec(learner) != before).any()

    def test_step_x(self, ensemble_voter):
        x = IO([torch.randn(8, 3)])
        t = IO([torch.randn(8, 4)])
        state = State()
        ensemble_voter.forward_io(x, state)
        ensemble_voter.accumulate(x, t, state)
        result = ensemble_voter.step_x(x, t, state)
        assert isinstance(result, IO)


class TestEnsembleLearner:

    def test_forward_nn(self, ensemble):
        x = IO([torch.randn(8, 3)])
        state = State()
        result = ensemble.forward_nn(x, state)
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([8, 4])
        assert all(isinstance(r, torch.Tensor) for r in result)

    def test_forward_nn_spawned(self, ensemble):
        x = IO([torch.randn(8, 3)])
        state = State()
        ensemble.adv()
        result = ensemble.forward_nn(x, state)
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([8, 4])
        assert all(isinstance(r, torch.Tensor) for r in result)

    def test_forward_io(self, ensemble):
        x = IO([torch.randn(8, 3)])
        state = State()
        result = ensemble.forward_io(x, state)
        assert isinstance(result, IO)
        assert len(result) == 1 or len(result) == 3
        assert all(isinstance(r, torch.Tensor) for r in result)

    def test_step(self, ensemble):
        x = IO([torch.randn(8, 3)])
        t = IO([torch.randn(8, 4)])
        state = State()
        ensemble.adv()
        ensemble.forward_io(x, state)
        ensemble.accumulate(x, t, state)
        befores = []
        for learner in ensemble.learners:
            befores.append(utils.to_pvec(learner))
        ensemble.step(x, t, state)
        for before, learner in enumerate(ensemble.learners):
            assert (utils.to_pvec(learner) != before).any()

    def test_step_with_not_use_last(self, ensemble):
        x = IO([torch.randn(8, 3)])
        t = IO([torch.randn(8, 4)])
        state = State()
        ensemble.adv()
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

    def test_step_x_with_adv_and_no_use_last(self, ensembleb):
        x = IO([torch.randn(8, 3)])
        t = IO([torch.randn(8, 4)])
        state = State()
        ensembleb.adv()
        ensembleb.forward_io(x, state)
        ensembleb.accumulate(x, t, state)
        result = ensembleb.step_x(x, t, state)
        assert isinstance(result, IO)

    def test_step_x_and_no_use_last(self, ensembleb):
        x = IO([torch.randn(8, 3)])
        t = IO([torch.randn(8, 4)])
        state = State()
        ensembleb.forward_io(x, state)
        ensembleb.accumulate(x, t, state)
        result = ensembleb.step_x(x, t, state)
        assert isinstance(result, IO)
