import pytest
import torch

from zenkai import Assessment
from zenkai.tansaku.core import Individual, Population, select_best_individual, select_best_sample
from zenkai.utils import get_model_parameters

from .fixtures import (assessment1, binary_individual1, binary_individual2,
                       binary_x, binary_x2, individual1, individual2,
                       individual_model, model1, model2, p1, pop_x1, pop_x2,
                       population1, population1_with_assessment,
                       population2_with_assessment, x1, x2)


class TestIndividual:

    def test_get_returns_requested_parameter(self, x1, individual1):
        print(list(individual1.keys()))
        assert (x1 == individual1['x']).all()

    def test_set_parameter_updates_parameter(self, x1, individual1, p1):
        print(list(individual1.keys()))
        individual1.set_p(p1, "x")
        assert (p1.data == x1).all()
    
    def test_using_model_in_individual_retrieves_parameters(self, individual_model: Individual, model1, model2):
        individual_model.set_model(model2, "model")
        assert (get_model_parameters(model2) == get_model_parameters(model1)).all()

    def test_iter_returns_all_individuals(self, x1, x2, individual2):
        result = {k: v for k, v in individual2.items()}
        assert result['x1'] is x1
        assert result['x2'] is x2

    def test_report_sets_the_assessment(self, individual1, assessment1):
        individual1 = individual1.report(assessment1)
        assert individual1.assessment is assessment1


class TestPopulation:

    def test_get_returns_population_parameter(self, pop_x1, population1):
        assert population1['x'] is pop_x1
    
    def test_sets_returns_population_parameter_for_x(self, pop_x1, x1, population1):
        x1 = torch.clone(x1)
        assert (pop_x1[0] == population1["x", 0]).all()

    def test_cannot_set_if_sizes_are_not_the_same(self):
        with pytest.raises(ValueError):
            Population(x=torch.rand(3, 2, 2), y=torch.rand(2, 3, 3))

    def test_len_is_correct(self):
        population = Population(x=torch.rand(3, 2, 2), y=torch.rand(3, 3, 3))
        assert population.k == 3

    def test_loop_over_population_returns_three_individuals(self):
        population = Population(x=torch.rand(3, 2, 2), y=torch.rand(3, 3, 3))
        individuals = list(population.individuals())
        assert len(individuals) == 3
        assert isinstance(individuals[0], Individual)

    def test_set_model_sets_the_model_correctly(self):
        linear = torch.nn.Linear(3, 3)
        population = Population(x=torch.rand(3, 2, 2), y=torch.rand(3, 12))
        population.set_model(linear, "y", 0)
        assert (
            get_model_parameters(linear) == population["y", 0]
        ).all()

    def test_set_p_sets_parameters_correctly(self, p1, pop_x1):
        population = Population(x=pop_x1)
        population.set_p(p1, "x", 1)
        assert (p1 == pop_x1[1]).all()

    def test_gather_sub_gets_correct_output_when_dim_is_3(self):

        x1 = torch.rand(4, 2, 2)
        gather_by = torch.randint(0, 4, (6, 2))

        population = Population(x=x1)
        gathered = population.gather_sub(gather_by)
        assert gathered['x'].size() == torch.Size([6, 2, 2])

    def test_gather_sub_gets_correct_output_when_dim_is_2(self):

        x1 = torch.rand(4, 2)
        gather_by = torch.randint(0, 4, (6, 2))

        population = Population(x=x1)
        gathered = population.gather_sub(gather_by)
        assert gathered['x'].size() == torch.Size([6, 2])

    def test_gather_sub_gets_correct_output_when_dim_is_4(self):

        x1 = torch.rand(4, 2, 2, 3)
        gather_by = torch.randint(0, 4, (6, 2))

        population = Population(x=x1)
        gathered = population.gather_sub(gather_by)
        assert gathered['x'].size() == torch.Size([6, 2, 2, 3])

    def test_gather_sub_raises_error_if_dim_too_large(self):

        x1 = torch.rand(4, 2)
        gather_by = torch.randint(0, 4, (6, 2, 2))

        population = Population(x=x1)
        with pytest.raises(ValueError):
            population.gather_sub(gather_by)


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
