from zenkai.tansaku.core import Individual, Population
from .fixtures import (
    pop_x1, population1, individual1, individual2, 
    individual_model, model1, model2, pop_x2, population1_with_assessment, population2_with_assessment,
    x1, x2, binary_x2, binary_individual1, binary_x, binary_individual2, binary_population2_with_assessment
)
from zenkai.utils import get_model_parameters
from zenkai.tansaku.selectors import BestSelectorIndividual, BestSelectorFeature, MomentumSelector, SlopeSelector, BinaryGaussianSelector


class TestBestSelector:
    
    def test_best_selector_returns_best_with_one_dimensions(self, population1_with_assessment):

        selector = BestSelectorIndividual()
        individual = selector(population1_with_assessment)
        assert individual['x'].size() == population1_with_assessment["x"].shape[1:]

    def test_best_selector_returns_best_with_two_dimensions(self, population2_with_assessment):

        selector = BestSelectorFeature()
        individual = selector(population2_with_assessment)
        assert individual['x'].size() == population2_with_assessment["x"].shape[1:]


class TestMomentumSelector:

    def test_momentum_selector_returns_best_with_one_dimensions(self, population1_with_assessment):

        selector = MomentumSelector(BestSelectorIndividual(), 0.1)
        individual = selector(population1_with_assessment)
        assert individual['x'].size() == population1_with_assessment["x"].shape[1:]

    def test_momentum_selector_returns_best_with_two_dimensions(self, population2_with_assessment):

        selector = MomentumSelector(BestSelectorFeature(), 0.1)
        individual = selector(population2_with_assessment)
        assert individual['x'].size() == population2_with_assessment["x"].shape[1:]

    def test_momentum_selector_returns_best_after_two_iterations_two_dimensions(self, population2_with_assessment):

        selector = MomentumSelector(BestSelectorFeature(), 0.1)
        individual = selector(population2_with_assessment)
        individual = selector(population2_with_assessment)
        assert individual['x'].size() == population2_with_assessment["x"].shape[1:]


class TestSlopeSelector:

    def test_slope_selector_returns_slope_with_one_dimensions(self, population2_with_assessment):

        selector = SlopeSelector(0.1)
        individual = selector(population2_with_assessment)
        assert individual['x'].size() == population2_with_assessment["x"].shape[1:]

    def test_slope_selector_returns_slope_after_two_iterations(self, population2_with_assessment):

        selector = SlopeSelector(0.1)
        individual = selector(population2_with_assessment)
        individual = selector(population2_with_assessment)
        assert individual['x'].size() == population2_with_assessment["x"].shape[1:]


class TestBinaryGaussianSelector:

    def test_gaussian_selector_returns_slope_with_one_dimensions(self, binary_population2_with_assessment):

        selector = BinaryGaussianSelector('x')
        individual = selector(binary_population2_with_assessment)
        assert individual['x'].size() == binary_population2_with_assessment["x"].shape[1:]

    def test_gaussian_selector_returns_slope_after_two_iterations(self, binary_population2_with_assessment):

        selector = BinaryGaussianSelector('x')
        individual = selector(binary_population2_with_assessment)
        assert individual['x'].size() == binary_population2_with_assessment["x"].shape[1:]
