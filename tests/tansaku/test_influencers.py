import torch

from zenkai import Assessment
from zenkai.tansaku.core import Individual, Population
from zenkai.tansaku.influencers import (BinaryAdjGaussianInfluencer,
                                      BinaryGaussianInfluencer,
                                      BinaryProbInfluencer, SlopeInfluencer)
from zenkai.utils import get_model_parameters

from .fixtures import (binary_individual1, binary_individual2, binary_x,
                       binary_x2, individual1, individual2, individual_model,
                       model1, model2, pop_x1, pop_x2, population1,
                       population1_with_assessment,
                       population2_with_assessment, x1, x2)


class TestSlopeInfluencer:

    def test_slope_modifier_returns_slope_with_one_update(self, population2_with_assessment):

        individual = Individual(x=torch.rand(population2_with_assessment["x"][0].size()))
        modiifer = SlopeInfluencer(0.1, 0.1)
        individual = modiifer(individual, population2_with_assessment)
        assert individual['x'].size() == population2_with_assessment["x"].shape[1:]

    def test_slope_modifier_returns_slope_after_two_iterations(self, population2_with_assessment):
        
        individual = Individual(x=torch.rand(population2_with_assessment["x"][0].size()))
        modifier = SlopeInfluencer(0.1, 0.1)
        individual = modifier(individual, population2_with_assessment)
        individual = modifier(individual, population2_with_assessment)
        assert individual['x'].size() == population2_with_assessment["x"].shape[1:]


class TestBinaryProbInfluencer:

    def test_binary_prob_modifier_outputs_neg_one_or_one(self):

        individual = Individual(x=torch.randn(2, 3).sign())
        population = Population(
            x=torch.randn(3, 2, 3).sign(), t=torch.randn(3, 2, 4).sign()
        ).report(
            Assessment(torch.randn(3, 2, 4))
        )

        modifier = BinaryProbInfluencer(False)
        result = modifier(individual, population)
        assert ((result["x"] == 1) | (result["x"] == -1)).all()

    def test_binary_prob_modifier_outputs_zero_or_one(self):

        individual = Individual(x=(torch.randn(2, 3) > 0).float())
        population = Population(
            x=(torch.rand(3, 2, 3) > 0).float(), t=(torch.rand(3, 2, 4) > 0).float()
        ).report(
            Assessment(torch.randn(3, 2, 4))
        )

        modifier = BinaryProbInfluencer(True)
        result = modifier(individual, population)
        assert ((result["x"] == 1) | (result["x"] == 0)).all()


class TestBinaryAdjInfluencer:

    def test_binary_adj_modifier_outputs_neg_one_or_one(self):

        individual = Individual(x=torch.randn(2, 3).sign())
        population = Population(
            x=torch.randn(8, 2, 3).sign(), t=torch.randn(8, 2, 4).sign()
        ).report(
            Assessment(torch.randn(8, 2))
        )

        modifier = BinaryAdjGaussianInfluencer(8, zero_neg=False)
        result = modifier(individual, population)
        assert ((result["x"] == 1) | (result["x"] == -1)).all()

    def test_binary_adj_modifier_outputs_zero_or_one(self):

        individual = Individual(x=(torch.randn(2, 3) > 0).float())
        population = Population(
            x=(torch.randn(8, 2, 3) > 0).float(), t=(torch.randn(8, 2, 3) > 0).float()
        ).report(
            Assessment(torch.randn(8, 2))
        )

        modifier = BinaryAdjGaussianInfluencer(8, zero_neg=True)
        result = modifier(individual, population)
        assert ((result["x"] == 1) | (result["x"] == 0)).all()


class TestBinaryGaussianInfluencer:

    def test_binary_modifier_returns_slope_with_one_update(self):

        individual = Individual(x=torch.randn(2, 3).sign())
        population = Population(
            x=torch.randn(8, 2, 3).sign(), t=torch.randn(8, 2, 4).sign()
        ).report(
            Assessment(torch.randn(8, 2))
        )

        modifier = BinaryGaussianInfluencer(8, zero_neg=False)
        result = modifier(individual, population)
        assert ((result["x"] == 1) | (result["x"] == -1)).all()

    def test_binary_modifier_returns_slope_with_zero_neg(self):

        individual = Individual(x=(torch.randn(2, 3) > 0).float())
        population = Population(
            x=(torch.randn(8, 2, 3) > 0).float(), t=(torch.randn(8, 2, 3) > 0).float()
        ).report(
            Assessment(torch.randn(8, 2))
        )

        modifier = BinaryGaussianInfluencer(8, zero_neg=True)
        result = modifier(individual, population)
        assert ((result["x"] == 1) | (result["x"] == 0)).all()
