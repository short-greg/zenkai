# 3rd party
import torch

# local
from zenkai import tansaku, Assessment
from zenkai.kaku import Individual, Population


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
