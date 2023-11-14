
from zenkai.tansaku._reduction import (
    BestSampleReducer,
    BestIndividualReducer,
    BinaryGaussianReducer,
    MomentumReducer,
)

import pytest
import torch

from zenkai import Assessment
from zenkai.tansaku._keep import Population


@pytest.fixture
def population1_with_assessment(pop_x1) -> Population:
    return Population(x=pop_x1).report(Assessment(torch.rand(pop_x1.size(0)).cumsum(0)))


@pytest.fixture
def population2_with_assessment(pop_x2) -> Population:
    return Population(x=pop_x2).report(
        Assessment(torch.rand(pop_x2.shape[:2]).cumsum(0))
    )


@pytest.fixture
def pop_x1():
    generator = torch.Generator()
    generator.manual_seed(1)
    return torch.rand(3, 2, 2, generator=generator)


@pytest.fixture
def pop_x2():
    generator = torch.Generator()
    generator.manual_seed(3)
    return torch.rand(3, 4, 2, generator=generator)


@pytest.fixture
def binary_population2_with_assessment(pop_x2) -> Population:
    return Population(x=pop_x2.sign()).report(
        Assessment(torch.rand(pop_x2.shape[:2]).cumsum(0))
    )


class TestBestReducer:
    def test_best_selector_returns_best_with_one_dimensions(
        self, population1_with_assessment
    ):

        selector = BestIndividualReducer()
        individual = selector(population1_with_assessment)
        assert individual["x"].size() == population1_with_assessment["x"].shape[1:]

    def test_best_selector_returns_best_with_two_dimensions(
        self, population2_with_assessment
    ):

        selector = BestSampleReducer()
        individual = selector(population2_with_assessment)
        assert individual["x"].size() == population2_with_assessment["x"].shape[1:]


class TestMomentumReducer:
    def test_momentum_selector_returns_best_with_one_dimensions(
        self, population1_with_assessment
    ):

        selector = MomentumReducer(BestIndividualReducer(), 0.1)
        individual = selector(population1_with_assessment)
        assert individual["x"].size() == population1_with_assessment["x"].shape[1:]

    def test_momentum_selector_returns_best_with_two_dimensions(
        self, population2_with_assessment
    ):

        selector = MomentumReducer(BestSampleReducer(), 0.1)
        individual = selector(population2_with_assessment)
        assert individual["x"].size() == population2_with_assessment["x"].shape[1:]

    def test_momentum_selector_returns_best_after_two_iterations_two_dimensions(
        self, population2_with_assessment
    ):

        selector = MomentumReducer(BestSampleReducer(), 0.1)
        individual = selector(population2_with_assessment)
        individual = selector(population2_with_assessment)
        assert individual["x"].size() == population2_with_assessment["x"].shape[1:]


class TestBinaryGaussianReducer:
    def test_gaussian_selector_returns_slope_with_one_dimensions(
        self, binary_population2_with_assessment
    ):

        selector = BinaryGaussianReducer("x")
        individual = selector(binary_population2_with_assessment)
        assert (
            individual["x"].size() == binary_population2_with_assessment["x"].shape[1:]
        )

    def test_gaussian_selector_returns_slope_after_two_iterations(
        self, binary_population2_with_assessment
    ):

        selector = BinaryGaussianReducer("x")
        individual = selector(binary_population2_with_assessment)
        assert (
            individual["x"].size() == binary_population2_with_assessment["x"].shape[1:]
        )
