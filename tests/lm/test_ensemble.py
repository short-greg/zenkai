import torch
from torch import nn
from zenkai.lm._io2 import IO as IO, iou
from zenkai.lm._ensemble import EnsembleLearner
from zenkai.nnz._ensemble_mod import StochasticVoter
from .test_grad import THGradLearnerT1
from zenkai.lm._state import State


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


class DummyEnsembleLearner(EnsembleLearner):

    def __init__(self, in_features: int, out_features: int, n_learners: int):

        super().__init__()
        self._learners = [
            THGradLearnerT1(in_features, out_features) for i in range(n_learners)
        ]

    def vote(self, x: IO, state: State) -> IO:

        votes = [learner.forward_io(x, state.sub(i)).f for i, learner in enumerate(self._learners)]
        return iou(torch.stack(votes))

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        return torch.tensor(1)

    def reduce(self, x: IO, state: State) -> IO:
        return iou(torch.mean(x.f, dim=0))

    def step(self, x: IO, t: IO, state: State):

        for i, learner in enumerate(self._learners):
            learner.step(x, t, state.sub(i))

    def step_x(self, x: IO, t: IO, state: State) -> IO:

        x_primes = []
        for i, learner in enumerate(self._learners):
            x_primes.append(learner.step(x, t, state.sub(i)).f)
        return iou(sum(x_primes))


class TestEnsembleLearnerVoter:
    def test_forward_produces_result_of_correct_shape(self):

        state = State()
        learner = DummyEnsembleLearner(2, 3, 3)
        assert learner.forward_io(iou(torch.rand(3, 2)), state).f.shape == torch.Size([3, 3])

    def test_vote_results_in_three_votes(self):

        state = State()
        learner = DummyEnsembleLearner(2, 3, 3)
        assert learner.vote(iou(torch.rand(4, 2)), state).f.size(0) == 3
