# 3rd party
import torch

# local
from zenkai import tansaku, Assessment
from zenkai.kaku import Individual, Population

import torch

from zenkai import Assessment
from zenkai.kaku import Population
from zenkai.tansaku import _genetic, _select


class TestKeepMixer:

    def test_keep_mixer_results_in_correct_size(self):

        individual1 = Individual(x=torch.rand(4, 2))
        individual2 = Individual(x=torch.rand(4, 2))
        new_individual = tansaku.keep_mixer(individual1, individual2, 0.9)

        assert new_individual["x"].shape == individual1["x"].shape

    def test_keep_mixer_results_in_same_as_original(self):

        individual1 = Individual(x=torch.rand(4, 2))
        individual2 = Individual(x=torch.rand(4, 2))
        new_individual = tansaku.keep_mixer(individual1, individual2, 1.0)
        assert (new_individual["x"] == individual1["x"]).all()

    def test_keep_mixer_results_in_same_as_new(self):

        individual1 = Individual(x=torch.rand(4, 2))
        individual2 = Individual(x=torch.rand(4, 2))
        new_individual = tansaku.keep_mixer(individual1, individual2, 0.0)
        assert (new_individual["x"] == individual2["x"]).all()


class TestBinaryRandCrossOver:

    def test_binary_rand_crossover(self):

        mixer = tansaku.BinaryRandCrossOver(0.5)
        population1 = Population(x=torch.randn(4, 4, 2).sign())
        population2 = Population(x=torch.randn(4, 4, 2).sign())
        new_population = mixer(population1, population2)
        assert ((new_population["x"] == -1) | (new_population["x"] == 1)).all()

    def test_binary_rand_crossover_with_real_values(self):

        mixer = tansaku.BinaryRandCrossOver(0.5)
        population1 = Population(x=torch.randn(4, 4, 2))
        population2 = Population(x=torch.randn(4, 4, 2))
        new_population = mixer(population1, population2)
        assert (
            (new_population["x"] == population1["x"])
            | (new_population["x"] == population2["x"])
        ).all()


class TestSmoothCrossOver:
    def test_gaussian_rand_crossover(self):

        mixer = tansaku.SmoothCrossOver()
        population1 = Population(x=torch.rand(4, 4, 2))
        population1.report(Assessment(torch.tensor([0.1, 0.2, 0.8, 1.0])))
        population2 = Population(x=torch.rand(4, 4, 2))
        new_population = mixer(population1, population2)
        assert new_population["x"].shape == population1["x"].shape

    def test_rand_crossover_produces_values_in_between(self):

        mixer = tansaku.SmoothCrossOver()
        population1 = Population(x=torch.randn(4, 4, 2).sign())
        population2 = Population(x=torch.randn(4, 4, 2).sign())
        new_population = mixer(population1, population2)
        assert (
            (new_population["x"] >= population1["x"])
            & (new_population["x"] <= population2["x"])
            | (new_population["x"] <= population1["x"])
            & (new_population["x"] >= population2["x"])
        ).all()


class TestProbDivider:

    def test_divides_into_correct_sizes(self):

        torch.manual_seed(1)
        population = Population(x=torch.rand(8, 4, 2))
        population.report(
            Assessment(torch.tensor([0.1, 0.4, 0.3, 0.2, 0.8, 1.0, 0.2, 1.0]))
        )

        divider = _genetic.Divider(_select.ProbSelector(3, _select.ToFitnessProb(0), c=2))
        child1, child2 = divider(population)
        assert child1['x'].shape == torch.Size([3, 4, 2])
        assert child1.k == 3

        assert child2.k == 3

    def test_divides_into_correct_sizes_with_rank(self):

        torch.manual_seed(1)
        population = Population(x=torch.rand(8, 4, 2))
        population.report(
            Assessment(torch.tensor([0.1, 0.4, 0.3, 0.2, 0.8, 1.0, 0.2, 1.0]))
        )

        divider = _genetic.Divider(_select.ProbSelector(3, _select.ToRankProb(0), c=2))
        child1, child2 = divider(population)
        assert child1['x'].shape == torch.Size([3, 4, 2])
        assert child1.k == 3
        assert child2.k == 3

    def test_divides_into_correct_sizes_with_two_dims_for_assessment(self):

        torch.manual_seed(1)
        population = Population(x=torch.rand(8, 4, 2))
        population.report(Assessment(torch.rand(8, 4)))

        divider = _genetic.Divider(_select.ProbSelector(3, _select.ToRankProb(0), c=2))
        child1, child2 = divider(population)

        assert child1.k == 3
        assert child2.k == 3
        assert child1['x'].shape == torch.Size([3, 4, 2])

    def test_divides_into_correct_sizes_when_div_start_is_two(self):

        torch.manual_seed(1)
        population = Population(x=torch.rand(8, 4, 2))
        population.report(Assessment(torch.rand(8, 4)))

        divider = _genetic.Divider(_select.ProbSelector(3, _select.ToFitnessProb(1), c=2))
        child1, child2 = divider(population)
        assert child1['x'].shape == torch.Size([8, 3, 2])
        assert child1.k == 8
        assert child2.k == 8


class TestKBestElitism:
    def test_k_best_elitism_chooses_best_with_minimize(self):

        torch.manual_seed(1)

        mixer = _genetic.Elitism(_select.TopKSelector(2, 0))
        population1 = Population(x=torch.rand(4, 4, 2))
        population1.report(Assessment(torch.tensor([0.1, 0.2, 0.8, 1.0])))
        population2 = Population(x=torch.rand(4, 4, 2))
        new_population = mixer(population1, population2)
        print(population1, new_population)
        assert (new_population.sub[[0, 1]]["x"] == population1.sub[[0, 1]]["x"]).all()
        assert (new_population.sub[[2, 3, 4, 5]]["x"] == population2["x"]).all()

    def test_k_best_elitism_chooses_best_with_maximize(self):

        torch.manual_seed(1)
        mixer = _genetic.Elitism(_select.TopKSelector(2, 0))
        population1 = Population(x=torch.rand(4, 4, 2))
        population1.report(Assessment(torch.tensor([0.1, 0.2, 0.8, 1.0]), True))
        population2 = Population(x=torch.rand(4, 4, 2))
        new_population = mixer(population1, population2)
        assert (new_population.sub[[0, 1]]["x"] == population1.sub[[3, 2]]["x"]).all()
        assert (new_population.sub[[2, 3, 4, 5]]["x"] == population2["x"]).all()


#     def test_divides_into_correct_sizes_when_div_start_is_two_and_dim_is_1_with_rank(
#         self,
#     ):

#         torch.manual_seed(1)
#         population = Population(x=torch.rand(8, 4))
#         population.report(Assessment(torch.rand(8, 4)))

#         divider = dividers.ProbDivider(_select.RankParentSelector(3), 2)
#         child1, child2 = divider(population)
#         assert child1.k == 3
#         assert child2.k == 3

#     def test_raises_error_if_negative_assessments(self):
#         torch.manual_seed(1)

#         population = Population(x=torch.rand(8, 4, 2))
#         population.report(Assessment(torch.randn(8, 4)))

#         divider = dividers.ProbDivider(_select.FitnessParentSelector(3))
#         with pytest.raises(ValueError):
#             divider(population)


# class TestFitnessEqualDivider:
#     def test_divides_into_correct_sizes(self):

#         torch.manual_seed(1)
#         population = Population(x=torch.rand(8, 4, 2))
#         population.report(
#             Assessment(torch.tensor([0.1, 0.4, 0.3, 0.2, 0.8, 1.0, 0.2, 1.0]))
#         )

#         divider = dividers.EqualDivider()
#         child1, child2 = divider(population)
#         assert child1.k == 8
#         assert child2.k == 8

# #     def test_divides_into_correct_sizes_with_two_dims_for_assessment(self):

# #         torch.manual_seed(1)
# #         population = Population(x=torch.rand(8, 4, 2))
# #         population.report(Assessment(torch.rand(8, 4)))

# #         divider = dividers.EqualDivider()
# #         child1, child2 = divider(population)
# #         assert child1.k == 8
# #         assert child2.k == 8

# # 3rd party
# import torch

# # local
# from zenkai import Assessment
# from zenkai import kaku
# from zenkai.tansaku import _elitism

