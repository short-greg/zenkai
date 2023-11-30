import pytest
import torch

from zenkai import Assessment
from zenkai.tansaku._select import (
    select_best_individual,
    select_best_sample,
    TopKSelector,
    BestSelector,
    RankParentSelector,
    FitnessParentSelector,
)
from zenkai import Individual, Population


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


class TestSelectBestSample:
    def test_select_best_sample_returns_best_sample(self):

        x = torch.tensor([[[0, 1], [0, 2]], [[1, 0], [2, 0]]])
        value = Assessment(torch.tensor([[0.1, 2.0], [2.0, 0.05]]))
        samples = select_best_sample(x, value)
        assert (samples[0] == x[0, 0]).all()
        assert (samples[1] == x[1, 1]).all()

    def test_select_best_sample_returns_best_sample_with_maximize(self):

        x = torch.tensor([[[0, 1], [0, 2]], [[1, 0], [2, 0]]])
        value = Assessment(torch.tensor([[0.1, 2.0], [2.0, 0.05]]), maximize=True)
        samples = select_best_sample(x, value)
        assert (samples[0] == x[1, 0]).all()
        assert (samples[1] == x[0, 1]).all()

    def test_select_best_raises_value_error_if_assessment_size_is_wrong(self):

        x = torch.tensor([[[0, 1], [0, 2]], [[1, 0], [2, 0]]])
        value = Assessment(torch.tensor([0.1, 0.2]))
        with pytest.raises(ValueError):
            select_best_sample(x, value)


class TestSelectBestIndividual:
    def test_select_best_individual_raises_value_error_with_incorrect_assessment_dim(
        self,
    ):

        x = torch.tensor([[[0, 1], [0, 2]], [[1, 0], [2, 0]]])
        value = Assessment(torch.tensor([[0.1, 2.0], [2.0, 0.05]]))
        with pytest.raises(ValueError):
            select_best_individual(x, value)

    def test_select_best_individual_returns_best_individual(self):

        x = torch.tensor([[[0, 1], [0, 2]], [[1, 0], [2, 0]]])
        value = Assessment(torch.tensor([0.1, 2.0]))
        samples = select_best_individual(x, value)
        assert (samples[0] == x[0, 0]).all()
        assert (samples[1] == x[0, 1]).all()

    def test_select_best_individual_returns_best_individual_with_maximize(self):

        x = torch.tensor([[[0, 1], [0, 2]], [[1, 0], [2, 0]]])
        value = Assessment(torch.tensor([0.1, 2.0]), maximize=True)
        samples = select_best_individual(x, value)
        assert (samples[0] == x[1, 0]).all()
        assert (samples[1] == x[1, 1]).all()


class TestTopKSelector:
    def test_select_retrieves_the_two_best(self):

        x = torch.tensor([[0, 1], [0, 2], [2, 3]])
        t = torch.tensor([[2, 3], [0, 2]])
        value = Assessment(torch.tensor([0.1, 2.0, 3.0]), maximize=True)
        selector = TopKSelector(2, dim=0)
        index_map = selector.select(value)
        assert (index_map.select_index(Individual(x=x))["x"] == t).all()

    def test_select_retrieves_the_two_best_for_dim2(self):

        x = torch.tensor([[0, 1], [0, 2], [2, 3]])
        t = torch.tensor([[1], [0], [2]])
        value = Assessment(
            torch.tensor([[0.1, 0.2], [2.0, 0.3], [3.0, 0.1]]), maximize=True
        )
        selector = TopKSelector(1, dim=1)
        index_map = selector.select(value)
        assert (index_map.select_index(Individual(x=x))["x"] == t).all()

    def test_select_retrieves_correct_shape_for_three_dims(self):

        x = torch.randn(3, 3, 3)
        value = Assessment(torch.randn(3, 3), maximize=True)
        selector = TopKSelector(2, dim=1)
        index_map = selector.select(value)
        assert index_map.select_index(Individual(x=x))["x"].shape == torch.Size(
            [3, 2, 3]
        )


class TestBestSelector:

    def test_select_retrieves_the_two_best(self):

        x = torch.tensor([[0, 1], [0, 2], [2, 3]])
        t = torch.tensor([[2, 3]])
        value = Assessment(torch.tensor([0.1, 2.0, 3.0]), maximize=True)
        selector = BestSelector(dim=0)
        index_map = selector.select(value)
        assert (index_map.select_index(Individual(x=x))["x"] == t).all()

    def test_select_retrieves_the_two_best_for_dim2(self):

        x = torch.tensor([[0, 1], [0, 2], [2, 3]])
        t = torch.tensor([[1], [0], [2]])
        value = Assessment(
            torch.tensor([[0.1, 0.2], [2.0, 0.3], [3.0, 0.1]]), maximize=True
        )
        selector = BestSelector(dim=1)
        index_map = selector.select(value)
        assert (index_map.select_index(Individual(x=x))["x"] == t).all()


class TestFitnessParentSelector:
    def test_select_retrieves_two_parents_of_length_two(self):

        x = torch.tensor([[0, 1], [0, 2], [2, 3]])
        value = Assessment(torch.tensor([0.1, 2.0, 3.0]), maximize=True)
        selector = FitnessParentSelector(k=2)
        index_map = selector.select(value)
        parent1, parent2 = index_map.select_index(Individual(x=x))
        assert parent1["x"].shape == torch.Size([2, 2])
        assert parent2["x"].shape == torch.Size([2, 2])


class TestRankParentSelector:
    def test_select_retrieves_two_parents_of_length_two(self):

        x = torch.tensor([[0, 1], [0, 2], [2, 3]])
        value = Assessment(torch.tensor([0.1, 2.0, 3.0]), maximize=True)
        selector = RankParentSelector(k=2)
        index_map = selector.select(value)
        parent1, parent2 = index_map.select_index(Individual(x=x))
        assert parent1["x"].shape == torch.Size([2, 2])
        assert parent2["x"].shape == torch.Size([2, 2])
