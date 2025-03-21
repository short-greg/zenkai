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

    # def test_adv(self, ensemble):
    #     initial_count = len(ensemble.learners)
    #     ensemble.adv()
    #     assert len(ensemble.learners) == initial_count + 1 or len(ensemble.learners) == ensemble._n_votes


# class TestVoterPopulator:

#     def test_voter_populator_populates_correct_count(self):

#         voter = StochasticVoter(nn.Sequential(nn.Dropout(0.2), nn.Linear(4, 3)), 6)
#         populator = VoterPopulator(voter, "x")
#         individual = torch.randn(4, 4)
#         population = populator(individual)
#         assert population["x"].shape == torch.Size([6, 4, 3])

#     def test_votes_are_different_for_items_in_population(self):

#         voter = StochasticVoter(nn.Sequential(nn.Dropout(0.2), nn.Linear(4, 3)), 6)
#         populator = VoterPopulator(voter, "x")
#         individual = Individual(x=torch.randn(4, 4))
#         population = populator(individual)
#         assert (population["x"][0] != population["x"][1]).any()


# class DummyEnsembleLearner(EnsembleLearner):

#     def __init__(self, in_features: int, out_features: int, n_learners: int):

#         super().__init__()
#         self._learners = [
#             THGradLearnerT1(in_features, out_features) for i in range(n_learners)
#         ]

#     def vote(self, x: IO, state: State) -> IO:

#         votes = [learner.forward_io(x, state.sub(i)).f for i, learner in enumerate(self._learners)]
#         return iou(torch.stack(votes))

#     def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
#         return torch.tensor(1)

#     def reduce(self, x: IO, state: State) -> IO:
#         return iou(torch.mean(x.f, dim=0))

#     def step(self, x: IO, t: IO, state: State):

#         for i, learner in enumerate(self._learners):
#             learner.step(x, t, state.sub(i))

#     def step_x(self, x: IO, t: IO, state: State) -> IO:

#         x_primes = []
#         for i, learner in enumerate(self._learners):
#             x_primes.append(learner.step(x, t, state.sub(i)).f)
#         return iou(sum(x_primes))


# class TestEnsembleLearnerVoter:
#     def test_forward_produces_result_of_correct_shape(self):

#         state = State()
#         learner = DummyEnsembleLearner(2, 3, 3)
#         assert learner.forward_io(iou(torch.rand(3, 2)), state).f.shape == torch.Size([3, 3])

#     def test_vote_results_in_three_votes(self):

#         state = State()
#         learner = DummyEnsembleLearner(2, 3, 3)
#         assert learner.vote(iou(torch.rand(4, 2)), state).f.size(0) == 3

