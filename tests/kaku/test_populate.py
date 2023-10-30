import pytest
import torch

from zenkai.kaku import Individual, Population
from zenkai.utils import get_model_parameters

from .fixtures import (assessment1, binary_individual1, binary_individual2,
                       binary_x, binary_x2, individual1, individual2,
                       individual_model, model1, model2, p1, pop_x1, pop_x2,
                       population1, population1_with_assessment,
                       population2_with_assessment, x1, x2)


class TestIndividual:

    def test_get_returns_requested_parameter(self, x1, individual1):
        assert (x1 == individual1['x']).all()

    def test_set_parameter_updates_parameter(self, x1, individual1, p1):
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

    def test_add_individuals_adds_correctly(self, x1, x2, individual2):
        individual3 = individual2 + individual2
        assert (individual3['x1'] == (x1 + x1)).all()
        assert (individual3['x2'] == (x2 + x2)).all()

    def test_add_individuals_adds_correctly_when_not_exists(self, x1, x2, individual1, individual2):
        individual1 = individual1.copy()
        individual1['x1'] = individual1['x']
        individual3 = individual1 + individual2
        
        assert (individual3['x1'] == (x1 + x1)).all()
        assert (individual3['x2'] == x2).all()

    def test_sub_individuals_subs_correctly_when_not_exists(self, x1, x2, individual1, individual2):
        individual1 = individual1.copy()
        individual1['x1'] = x2
        individual3 = individual1 - individual2
        
        assert (individual3['x1'] == (x2 - x1)).all()
        assert individual3.get('x2') is None

    def test_sub_individuals_subs_correctly_when_not_exists(self, x1, x2, individual1, individual2):
        individual1 = individual1.copy()
        individual1['x1'] = x2
        individual3 = individual1 - individual2
        
        assert (individual3['x1'] == (x2 - x1)).all()
        assert individual3.get('x2') is None

    def test_mul_individuals_muls_correctly_when_not_exists(self, x1, x2, individual1, individual2):
        individual1 = individual1.copy()
        individual1['x1'] = x2
        individual3 = individual1 * individual2
        
        assert (individual3['x1'] == (x2 * x1)).all()
        assert individual3.get('x2') is None

    def test_le_gets_less_value_from_inviduals(self, individual1):
        individual1 = individual1.copy()
        individual2 = individual1 + 1
        individual3 = individual1 < individual2
        
        assert (individual3['x']).all()

    def test_lt_gets_less_value_from_inviduals(self, individual1):
        individual1 = individual1.copy()
        individual2 = individual1.copy()
        individual3 = individual1 <= individual2
        
        assert (individual3['x']).all()

    def test_gt_gets_greater_value_from_inviduals(self, individual1):
        individual1 = individual1.copy()
        individual2 = individual1.copy()
        individual3 = individual1 >= individual2
        
        assert (individual3['x']).all()

    def test_gt_gets_greater_value_from_inviduals(self, individual1):
        individual1 = individual1.copy()
        individual2 = individual1 - 1
        individual3 = individual1 > individual2
        
        assert (individual3['x']).all()

    def test_eq_returns_equal_if_same(self, individual1):
        individual1 = individual1.copy()
        individual2 = individual1.copy()
        
        assert (individual1 == individual2)['x'].all()


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

    def test_lt_gets_greater_value_from_inviduals(self):
        population = Population(x=torch.rand(3, 2, 2))
        population2 = Population(x=population['x']+0.1)
        assert (population < population2)['x'].all()
    
    def test_apply_rand_like_produces_similar_population(self):
        population = Population(x=torch.rand(3, 2, 2))
        population2 = population.apply(torch.rand_like)
        
        assert (population2['x'] != population['x']).any()
    
    def test_apply_add_produces_population_plus_1(self):
        population = Population(x=torch.rand(3, 2, 2))
        population2 = population.apply(lambda x: torch.add(x, 1))
        
        assert (population2['x'] == population['x'] + 1).all()

    def test_apply_add_does_not_add_if_filtered(self):
        population = Population(x=torch.rand(3, 2, 2))
        population2 = population.apply(lambda x: torch.add(x, 1), keys=['y'])
        
        assert (population2['x'] == population['x']).all()

    def test_apply_add_does_not_add_if_not_filtered(self):
        population = Population(x=torch.rand(3, 2, 2))
        population2 = population.apply(lambda x: torch.add(x, 1), keys=['x'])
        
        assert (population2['x'] == population['x'] + 1).all()
