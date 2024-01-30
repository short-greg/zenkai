import torch
import pytest

from zenkai import Individual, Population, Assessment
from zenkai.tansaku._manipulate import SlopeUpdater, SlopeCalculator, ApplyMomentum


@pytest.fixture
def x1():
    generator = torch.Generator()
    generator.manual_seed(1)
    return torch.rand(2, 2, generator=generator)


@pytest.fixture
def binary_x():
    generator = torch.Generator()
    generator.manual_seed(1)
    return torch.rand(2, 2, generator=generator).sign()


@pytest.fixture
def binary_x2():
    generator = torch.Generator()
    generator.manual_seed(1)
    return (torch.rand(2, 2, generator=generator) > 0.5).float()


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
def x2():
    generator = torch.Generator()
    generator.manual_seed(2)
    return torch.rand(2, 2, generator=generator)


@pytest.fixture
def model1():
    return torch.nn.Linear(2, 3)


@pytest.fixture
def model2() -> torch.nn.Module:
    return torch.nn.Linear(2, 3)


@pytest.fixture
def p1() -> torch.nn.parameter.Parameter:
    generator = torch.Generator()
    generator.manual_seed(3)
    return torch.nn.parameter.Parameter(torch.rand(2, 2, generator=generator))


@pytest.fixture
def individual1(x1) -> Individual:
    return Individual(x=x1)


@pytest.fixture
def individual2(x1, x2) -> Individual:
    return Individual(x1=x1, x2=x2)


@pytest.fixture
def binary_individual1(binary_x) -> Individual:
    return Individual(x=binary_x)


@pytest.fixture
def binary_individual2(binary_x2) -> Individual:
    return Individual(x=binary_x2)


@pytest.fixture
def individual_model(model1) -> Individual:
    return Individual(model=model1)


@pytest.fixture
def assessment1() -> Assessment:
    return Assessment(torch.rand(2, 3))


@pytest.fixture
def population1(pop_x1) -> Population:
    return Population(x=pop_x1)


@pytest.fixture
def population1_with_assessment(pop_x1) -> Population:
    return Population(x=pop_x1).report(Assessment(torch.rand(pop_x1.size(0)).cumsum(0)))


@pytest.fixture
def population2_with_assessment(pop_x2) -> Population:
    return Population(x=pop_x2).report(
        Assessment(torch.rand(pop_x2.shape[:2]).cumsum(0))
    )


@pytest.fixture
def binary_population2_with_assessment(pop_x2) -> Population:
    return Population(x=pop_x2.sign()).report(
        Assessment(torch.rand(pop_x2.shape[:2]).cumsum(0))
    )


class TestSlopeInfluencer:
    def test_slope_modifier_returns_slope_with_one_update(
        self, population2_with_assessment
    ):

        individual = Individual(
            x=torch.rand(population2_with_assessment["x"][0].size())
        )
        modiifer = SlopeUpdater(0.1, 0.1)
        individual = modiifer(individual, population2_with_assessment)
        assert individual["x"].size() == population2_with_assessment["x"].shape[1:]

    def test_slope_modifier_returns_slope_after_two_iterations(
        self, population2_with_assessment
    ):

        individual = Individual(
            x=torch.rand(population2_with_assessment["x"][0].size())
        )
        modifier = SlopeUpdater(0.1, 0.1)
        individual = modifier(individual, population2_with_assessment)
        individual = modifier(individual, population2_with_assessment)
        assert individual["x"].size() == population2_with_assessment["x"].shape[1:]


class TestSlopeCalculator:
    
    def test_slope_selector_returns_slope_with_one_dimensions(
        self, population2_with_assessment
    ):

        selector = SlopeCalculator(0.1)
        individual = selector(population2_with_assessment)
        assert individual["x"].size() == population2_with_assessment["x"].shape[1:]

    def test_slope_selector_returns_slope_after_two_iterations(
        self, population2_with_assessment
    ):

        selector = SlopeCalculator(0.1)
        individual = selector(population2_with_assessment)
        individual = selector(population2_with_assessment)
        assert individual["x"].size() == population2_with_assessment["x"].shape[1:]


class TestMomentum:

    def test_momentum_selector_returns_best_with_one_dimensions(
        self, individual1
    ):

        momentum = ApplyMomentum(0.1)
        individual = momentum(individual1)
        assert individual["x"].size() == individual1["x"].shape

    def test_momentum_selector_returns_best_with_two_dimensions(
        self, individual1
    ):

        momentum = ApplyMomentum(0.1)
        individual = momentum(individual1)
        assert individual["x"].size() == individual1["x"].shape

    def test_momentum_selector_returns_best_after_two_iterations_two_dimensions(
        self, individual1
    ):

        momentum = ApplyMomentum(0.1)
        individual = momentum(individual1)
        individual = momentum(individual1)
        assert individual["x"].size() == individual1["x"].shape

