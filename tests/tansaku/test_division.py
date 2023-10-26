import torch
import pytest

from zenkai import Assessment
from zenkai.kaku import Population
from zenkai.tansaku import division as dividers, select


class TestFitnessProportionalDivider:

    def test_divides_into_correct_sizes(self):

        torch.manual_seed(1)
        population = Population(x=torch.rand(8, 4, 2))
        population.report(Assessment(torch.tensor([0.1, 0.4, 0.3, 0.2, 0.8, 1.0, 0.2, 1.0])))
        
        divider = dividers.ProbDivider(select.FitnessParentSelector(3))
        child1, child2 = divider(population)
        assert child1.k == 3
        assert child2.k == 3

    def test_divides_into_correct_sizes_with_rank(self):

        torch.manual_seed(1)
        population = Population(x=torch.rand(8, 4, 2))
        population.report(Assessment(torch.tensor([0.1, 0.4, 0.3, 0.2, 0.8, 1.0, 0.2, 1.0])))
        
        divider = dividers.ProbDivider(select.RankParentSelector(3))
        child1, child2 = divider(population)
        assert child1.k == 3
        assert child2.k == 3

    def test_divides_into_correct_sizes_with_two_dims_for_assessment(self):

        torch.manual_seed(1)
        population = Population(x=torch.rand(8, 4, 2))
        population.report(Assessment(torch.rand(8, 4)))
        
        divider = dividers.ProbDivider(select.FitnessParentSelector(3))
        child1, child2 = divider(population)

        assert child1.k == 3
        assert child2.k == 3
    
    def test_divides_into_correct_sizes_when_div_start_is_two(self):

        torch.manual_seed(1)
        population = Population(x=torch.rand(8, 4, 2))
        population.report(Assessment(torch.rand(8, 4)))
        
        divider = dividers.ProbDivider(select.FitnessParentSelector(3))
        child1, child2 = divider(population)
        assert child1.k == 3
        assert child2.k == 3

    def test_divides_into_correct_sizes_when_div_start_is_two_and_dim_is_1(self):

        torch.manual_seed(1)
        population = Population(x=torch.rand(8, 4))
        population.report(Assessment(torch.rand(8, 4)))
        
        divider = dividers.ProbDivider(select.FitnessParentSelector(3), 2)
        child1, child2 = divider(population)
        assert child1.k == 3
        assert child2.k == 3

    def test_divides_into_correct_sizes_when_div_start_is_two_and_dim_is_1_with_rank(self):

        torch.manual_seed(1)
        population = Population(x=torch.rand(8, 4))
        population.report(Assessment(torch.rand(8, 4)))
        
        divider = dividers.ProbDivider(select.RankParentSelector(3), 2)
        child1, child2 = divider(population)
        assert child1.k == 3
        assert child2.k == 3

    def test_divides_into_correct_sizes_when_div_start_is_two_and_dim_is_1_with_rank(self):

        torch.manual_seed(1)
        population = Population(x=torch.rand(8, 4))
        population.report(Assessment(torch.rand(8, 4)))
        
        divider = dividers.ProbDivider(select.RankParentSelector(3), 2)
        child1, child2 = divider(population)
        assert child1.k == 3
        assert child2.k == 3

    def test_raises_error_if_negative_assessments(self):
        torch.manual_seed(1)

        population = Population(x=torch.rand(8, 4, 2))
        population.report(Assessment(torch.randn(8, 4)))
        
        divider = dividers.ProbDivider(select.FitnessParentSelector(3))
        with pytest.raises(ValueError):
            divider(population)


class TestFitnessEqualDivider:

    def test_divides_into_correct_sizes(self):

        torch.manual_seed(1)
        population = Population(x=torch.rand(8, 4, 2))
        population.report(Assessment(torch.tensor([0.1, 0.4, 0.3, 0.2, 0.8, 1.0, 0.2, 1.0])))
        
        divider = dividers.EqualDivider()
        child1, child2 = divider(population)
        assert child1.k == 8
        assert child2.k == 8

    def test_divides_into_correct_sizes_with_two_dims_for_assessment(self):

        torch.manual_seed(1)
        population = Population(x=torch.rand(8, 4, 2))
        population.report(Assessment(torch.rand(8, 4)))
        
        divider = dividers.EqualDivider()
        child1, child2 = divider(population)
        assert child1.k == 8
        assert child2.k == 8
