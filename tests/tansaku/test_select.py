import pytest
import torch

from zenkai import Assessment
from zenkai.tansaku._select import select_best_individual, select_best_sample, TopKSelector, BestSelector, RankParentSelector, FitnessParentSelector
from zenkai.utils import get_model_parameters
from zenkai import Individual

from .fixtures import (assessment1, binary_individual1, binary_individual2,
                       binary_x, binary_x2, individual1, individual2,
                       individual_model, model1, model2, p1, pop_x1, pop_x2,
                       population1, population1_with_assessment,
                       population2_with_assessment, x1, x2)


class TestSelectBestSample:

    def test_select_best_sample_returns_best_sample(self):

        x = torch.tensor(
            [[[0, 1], [0, 2]], [[1, 0], [2, 0]]]
        )
        value = Assessment(
            torch.tensor([[0.1, 2.0], [2.0, 0.05]])
        )
        samples = select_best_sample(x, value)
        assert (samples[0] == x[0, 0]).all()
        assert (samples[1] == x[1, 1]).all()

    def test_select_best_sample_returns_best_sample_with_maximize(self):

        x = torch.tensor(
            [[[0, 1], [0, 2]], [[1, 0], [2, 0]]]
        )
        value = Assessment(
            torch.tensor([[0.1, 2.0], [2.0, 0.05]]), maximize=True
        )
        samples = select_best_sample(x, value)
        assert (samples[0] == x[1, 0]).all()
        assert (samples[1] == x[0, 1]).all()

    def test_select_best_raises_value_error_if_assessment_size_is_wrong(self):

        x = torch.tensor(
            [[[0, 1], [0, 2]], [[1, 0], [2, 0]]]
        )
        value = Assessment(
            torch.tensor([0.1, 0.2])
        )
        with pytest.raises(ValueError):
            select_best_sample(x, value)
    

class TestSelectBestIndividual:

    def test_select_best_individual_raises_value_error_with_incorrect_assessment_dim(self):

        x = torch.tensor(
            [[[0, 1], [0, 2]], [[1, 0], [2, 0]]]
        )
        value = Assessment(
            torch.tensor([[0.1, 2.0], [2.0, 0.05]])
        )
        with pytest.raises(ValueError):
            select_best_individual(x, value)

    def test_select_best_individual_returns_best_individual(self):

        x = torch.tensor(
            [[[0, 1], [0, 2]], [[1, 0], [2, 0]]]
        )
        value = Assessment(
            torch.tensor([0.1, 2.0])
        )
        samples = select_best_individual(x, value)
        assert (samples[0] == x[0, 0]).all()
        assert (samples[1] == x[0, 1]).all()


    def test_select_best_individual_returns_best_individual_with_maximize(self):

        x = torch.tensor(
            [[[0, 1], [0, 2]], [[1, 0], [2, 0]]]
        )
        value = Assessment(
            torch.tensor([0.1, 2.0]), maximize=True
        )
        samples = select_best_individual(x, value)
        assert (samples[0] == x[1, 0]).all()
        assert (samples[1] == x[1, 1]).all()


class TestTopKSelector:

    def test_select_retrieves_the_two_best(self):

        x = torch.tensor(
            [[0, 1], [0, 2], [2, 3]]
        )
        t = torch.tensor(
            [[2, 3], [0, 2]]
        )
        value = Assessment(
            torch.tensor([0.1, 2.0, 3.0]), maximize=True
        )
        selector = TopKSelector(2, dim=0)
        index_map = selector.select(value)
        print(index_map.select_index(Individual(x=x))['x'])
        assert (index_map.select_index(Individual(x=x))['x'] == t).all()

    def test_select_retrieves_the_two_best_for_dim2(self):

        x = torch.tensor(
            [[0, 1], [0, 2], [2, 3]]
        )
        t = torch.tensor(
            [[1], [0], [2]]
        )
        value = Assessment(
            torch.tensor([[0.1, 0.2], [2.0, 0.3], [3.0, 0.1]]), maximize=True
        )
        selector = TopKSelector(1, dim=1)
        index_map = selector.select(value)
        assert (index_map.select_index(Individual(x=x))['x'] == t).all()

    def test_select_retrieves_correct_shape_for_three_dims(self):

        x = torch.randn(3, 3, 3)
        value = Assessment(
            torch.randn(3, 3), maximize=True
        )
        selector = TopKSelector(2, dim=1)
        index_map = selector.select(value)
        assert (index_map.select_index(Individual(x=x))['x'].shape == torch.Size([3, 2, 3]))


class TestBestSelector:

    def test_select_retrieves_the_two_best(self):

        x = torch.tensor(
            [[0, 1], [0, 2], [2, 3]]
        )
        t = torch.tensor(
            [[2, 3]]
        )
        value = Assessment(
            torch.tensor([0.1, 2.0, 3.0]), maximize=True
        )
        selector = BestSelector(dim=0)
        index_map = selector.select(value)
        print(index_map.select_index(Individual(x=x))['x'])
        assert (index_map.select_index(Individual(x=x))['x'] == t).all()

    def test_select_retrieves_the_two_best_for_dim2(self):

        x = torch.tensor(
            [[0, 1], [0, 2], [2, 3]]
        )
        t = torch.tensor(
            [[1], [0], [2]]
        )
        value = Assessment(
            torch.tensor([[0.1, 0.2], [2.0, 0.3], [3.0, 0.1]]), maximize=True
        )
        selector = BestSelector(dim=1)
        index_map = selector.select(value)
        assert (index_map.select_index(Individual(x=x))['x'] == t).all()


class TestFitnessParentSelector:

    def test_select_retrieves_two_parents_of_length_two(self):

        x = torch.tensor(
            [[0, 1], [0, 2], [2, 3]]
        )
        value = Assessment(
            torch.tensor([0.1, 2.0, 3.0]), maximize=True
        )
        selector = FitnessParentSelector(k=2)
        index_map = selector.select(value)
        parent1, parent2 = index_map.select_index(Individual(x=x))
        assert (parent1['x'].shape == torch.Size([2, 2]))
        assert (parent2['x'].shape == torch.Size([2, 2]))


class TestRankParentSelector:

    def test_select_retrieves_two_parents_of_length_two(self):

        x = torch.tensor(
            [[0, 1], [0, 2], [2, 3]]
        )
        value = Assessment(
            torch.tensor([0.1, 2.0, 3.0]), maximize=True
        )
        selector = RankParentSelector(k=2)
        index_map = selector.select(value)
        parent1, parent2 = index_map.select_index(Individual(x=x))
        assert (parent1['x'].shape == torch.Size([2, 2]))
        assert (parent2['x'].shape == torch.Size([2, 2]))
