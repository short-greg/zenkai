import torch

from zenkai.tansaku.functional import Individual
from zenkai.tansaku.slope import (SlopeUpdater, SlopeCalculator)
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


class TestSlopeCalculator:

    def test_slope_selector_returns_slope_with_one_dimensions(self, population2_with_assessment):

        selector = SlopeCalculator(0.1)
        individual = selector(population2_with_assessment)
        assert individual['x'].size() == population2_with_assessment["x"].shape[1:]

    def test_slope_selector_returns_slope_after_two_iterations(self, population2_with_assessment):

        selector = SlopeCalculator(0.1)
        individual = selector(population2_with_assessment)
        individual = selector(population2_with_assessment)
        assert individual['x'].size() == population2_with_assessment["x"].shape[1:]

