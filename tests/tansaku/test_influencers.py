import torch

from zenkai.tansaku.core import Individual
from zenkai.tansaku.slope import (SlopeUpdater, PopulationLimiter)
from zenkai.utils import get_model_parameters

from .fixtures import (binary_individual1, binary_individual2, binary_x,
                       binary_x2, individual1, individual2, individual_model,
                       model1, model2, pop_x1, pop_x2, population1,
                       population1_with_assessment,
                       population2_with_assessment, x1, x2)


class TestSlopeInfluencer:

    def test_slope_modifier_returns_slope_with_one_update(self, population2_with_assessment):

        individual = Individual(x=torch.rand(population2_with_assessment["x"][0].size()))
        modiifer = SlopeUpdater(0.1, 0.1)
        individual = modiifer(individual, population2_with_assessment)
        assert individual['x'].size() == population2_with_assessment["x"].shape[1:]

    def test_slope_modifier_returns_slope_after_two_iterations(self, population2_with_assessment):
        
        individual = Individual(x=torch.rand(population2_with_assessment["x"][0].size()))
        modifier = SlopeUpdater(0.1, 0.1)
        individual = modifier(individual, population2_with_assessment)
        individual = modifier(individual, population2_with_assessment)
        assert individual['x'].size() == population2_with_assessment["x"].shape[1:]


class TestPopulationLimiter:

    def test_slope_modifier_returns_slope_after_two_iterations(self, population2_with_assessment):
        
        individual = Individual(x=torch.rand(population2_with_assessment["x"][0].size()))
        modifier = PopulationLimiter(torch.tensor([0], dtype=torch.int64))
        population = modifier(population2_with_assessment, individual)
        assert population['x'].size() == population2_with_assessment["x"].shape
