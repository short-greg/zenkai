import pytest
import torch

from zenkai import Assessment, Individual, Population
from zenkai.tansaku.utils import select_best_individual, select_best_sample
from zenkai.utils import get_model_parameters

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
