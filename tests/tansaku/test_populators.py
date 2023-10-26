import pytest

import torch
from torch import nn
from zenkai import Assessment
from zenkai.tansaku._keep import Individual
from zenkai.kikai._ensemble import VoterPopulator
from zenkai.utils import get_model_parameters
from .fixtures import (binary_individual1, binary_individual2, binary_x,
                       binary_x2, individual1, individual2, individual_model,
                       model1, model2, pop_x1, pop_x2, population1,
                       population1_with_assessment,
                       population2_with_assessment, x1, x2)
from zenkai.mod._ensemble import StochasticVoter


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
