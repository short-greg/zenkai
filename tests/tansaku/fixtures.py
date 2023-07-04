import pytest
import torch

from zenkai import Assessment
from zenkai.tansaku.core import Individual, Population
from zenkai.utils import get_model_parameters


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
    return torch.nn.parameter.Parameter(
        torch.rand(2, 2, generator=generator)
    )


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
    return Population(x=pop_x2).report(Assessment(torch.rand(pop_x2.shape[:2]).cumsum(0)))

@pytest.fixture
def binary_population2_with_assessment(pop_x2) -> Population:
    return Population(x=pop_x2.sign()).report(Assessment(torch.rand(pop_x2.shape[:2]).cumsum(0)))

@pytest.fixture
def individual2(x1, x2) -> Individual:
    return Individual(x1=x1, x2=x2)
