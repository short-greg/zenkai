import torch
from torch import nn
from zenkai.kaku import IO, State
from zenkai.kaku._assess import Assessment
from zenkai.tansaku._keep import Individual
from zenkai.kikai._ensemble import VoterPopulator, EnsembleLearner
from zenkai.utils import get_model_parameters
from zenkai.mod._ensemble import StochasticVoter
from .test_grad import THGradLearnerT1


class TestVoterPopulator:

    def test_voter_populator_populates_correct_count(self):

        voter = StochasticVoter(nn.Sequential(nn.Dropout(0.2), nn.Linear(4, 3)), 6)
        populator = VoterPopulator(voter, 'x')
        individual = Individual(x=torch.randn(4, 4))
        population = populator(individual)
        assert population['x'].shape == torch.Size([6, 4, 3])

    def test_votes_are_different_for_items_in_population(self):

        voter = StochasticVoter(nn.Sequential(nn.Dropout(0.2), nn.Linear(4, 3)), 6)
        populator = VoterPopulator(voter, 'x')
        individual = Individual(x=torch.randn(4, 4))
        population = populator(individual)
        assert (population['x'][0] != population['x'][1]).any()


class DummyEnsembleLearner(EnsembleLearner):

    def __init__(self, in_features: int, out_features: int, n_learners: int):

        super().__init__()
        self._learners = [
            THGradLearnerT1(in_features, out_features) for i in range(n_learners)
        ]

    def vote(self, x: IO, state: State, release: bool = True) -> IO:
        
        votes = [
            learner(x, state, release).f for learner in self._learners
        ]
        return IO(
            torch.stack(votes)
        )
    
    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return Assessment(torch.tensor(1))
    
    def reduce(self, x: IO, state: State, release: bool = True) -> IO:
        return IO(torch.mean(
            x.f, dim=0
        )).out(release)
    
    def step(self, x: IO, t: IO, state: State):
        
        for learner in self._learners:
            learner.step(x, t, state)

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        
        x_primes = []
        for learner in self._learners:
            x_primes.append(learner.step(x, t, state).f)
        return IO(sum(x_primes))


class TestEnsembleLearnerVoter:
    
    def test_forward_produces_result_of_correct_shape(self):

        learner = DummyEnsembleLearner(2, 3, 3)
        assert learner(IO(torch.rand(3, 2))).f.shape == torch.Size([3, 3])
    
    def test_vote_results_in_three_votes(self):

        learner = DummyEnsembleLearner(2, 3, 3)
        assert learner.vote(IO(torch.rand(4, 2)), State()).f.size(0) == 3
