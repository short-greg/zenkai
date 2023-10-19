import pytest

import torch
from torch import nn
from zenkai import Assessment
from zenkai.tansaku.core import Individual, Population
from zenkai.tansaku.populators import (BinaryPopulator, ConservativePopulator,
                                       GaussianPopulator, RepeatPopulator, VoterPopulator)
from zenkai.utils import get_model_parameters, StochasticVoter
from .fixtures import (binary_individual1, binary_individual2, binary_x,
                       binary_x2, individual1, individual2, individual_model,
                       model1, model2, pop_x1, pop_x2, population1,
                       population1_with_assessment,
                       population2_with_assessment, x1, x2)


class TestRepeatPopulator:

    def test_repeat_individual(self, individual1):

        repeat_populator = RepeatPopulator(5)
        population = repeat_populator(individual1)
        assert (population["x", 0] == population["x", 1]).all()
        assert (population.k == 5)
        assert (population["x", 0] == individual1["x"]).all()


class TestGaussianPopulator:

    def test_gaussian_populator_spawns_correct_number_of_individuals(self, individual1):
        gaussian_populator = GaussianPopulator(5, 0.5)
        population = gaussian_populator(individual1)
        assert population.k == 5

    def test_gaussian_populator_spawns_correct_number_of_individuals_with_equal_change(self, individual1):
        gaussian_populator = GaussianPopulator(5, 0.5, equal_change_dim=1)
        population = gaussian_populator(individual1)
        assert population.k == 5


class TestBinaryPopulator:

    def test_binary_populator_spawns_correct_number_of_individuals(self, binary_individual1):
        binary_populator = BinaryPopulator(4)
        population = binary_populator(binary_individual1)
        assert population.k == 4

    def test_binary_populator_spawns_correct_number_of_individuals_with_equal_change(self, binary_individual1):
        gaussian_populator = BinaryPopulator(4, 0.5, equal_change_dim=1)
        population = gaussian_populator(binary_individual1)
        assert ((population["x"] == -1) | (population["x"] == 1)).all()
        assert population.k == 4

    def test_binary_populator_spawns_correct_number_of_individuals_with_equal_change(self, binary_individual2):
        gaussian_populator = BinaryPopulator(4, 0.5, zero_neg=True)
        population = gaussian_populator(binary_individual2)
        assert ((population["x"] == 0) | (population["x"] == 1)).all()
        assert population.k == 4


class TestConservativePopulator:

    def test_gaussian_populator_spawns_correct_number_of_individuals(self, individual1):
        gaussian_populator = ConservativePopulator(GaussianPopulator(5, 0.5), 0.1)
        population = gaussian_populator(individual1)
        assert population.k == 5

    def test_gaussian_populator_raises_error_if_rate_is_invalid(self, individual1):
        with pytest.raises(ValueError):
            ConservativePopulator(GaussianPopulator(5, 0.5), -1)


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
