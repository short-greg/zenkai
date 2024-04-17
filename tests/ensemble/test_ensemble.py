import torch
from torch import nn
from zenkai.kaku import IO
from zenkai.ensemble._ensemble import EnsembleLearner
from zenkai.ensemble._ensemble_mod import StochasticVoter
from ..kaku.test_grad import THGradLearnerT1


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

    def vote(self, x: IO, release: bool = True) -> IO:

        votes = [learner(x, release).f for learner in self._learners]
        return IO(torch.stack(votes))

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        return torch.tensor(1)

    def reduce(self, x: IO, release: bool = True) -> IO:
        return IO(torch.mean(x.f, dim=0)).out(release)

    def step(self, x: IO, t: IO):

        for learner in self._learners:
            learner.step(x, t)

    def step_x(self, x: IO, t: IO) -> IO:

        x_primes = []
        for learner in self._learners:
            x_primes.append(learner.step(x, t).f)
        return IO(sum(x_primes))


class TestEnsembleLearnerVoter:
    def test_forward_produces_result_of_correct_shape(self):

        learner = DummyEnsembleLearner(2, 3, 3)
        assert learner(IO(torch.rand(3, 2))).f.shape == torch.Size([3, 3])

    def test_vote_results_in_three_votes(self):

        learner = DummyEnsembleLearner(2, 3, 3)
        assert learner.vote(IO(torch.rand(4, 2))).f.size(0) == 3
