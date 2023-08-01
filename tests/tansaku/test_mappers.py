

# 3rd party
import torch
import torch.nn as nn

# local
from zenkai import tansaku


class TestGaussianMapper:

    def test_gaussian_mapper_generates_gaussian_population_after_one(self):

        mapper = tansaku.GaussianMapper(
            4
        )
        population = tansaku.Population(
            x=torch.randn(8, 4)
        )
        child = mapper(population)
        assert child['x'].shape == torch.Size([4, 4])
    
    def test_gaussian_mapper_generates_gaussian_population_after_one_with_std_and_mu(self):

        mapper = tansaku.GaussianMapper(
            4, mu0=0.0, std0=1.0
        )
        population = tansaku.Population(
            x=torch.randn(8, 4)
        )
        child = mapper(population)
        assert child['x'].shape == torch.Size([4, 4])

    def test_gaussian_mapper_generates_gaussian_population_after_two_updates(self):

        mapper = tansaku.GaussianMapper(
            4
        )
        population = tansaku.Population(
            x=torch.randn(8, 4)
        )
        population2 = tansaku.Population(
            x=torch.randn(8, 4)
        )
        child = mapper(population)
        child = mapper(population2)
        assert child['x'].shape == torch.Size([4, 4])


class TestBinaryMapper:

    def test_binary_mapper_generates_population_after_one(self):

        mapper = tansaku.BinaryMapper(
            4
        )
        population = tansaku.Population(
            x=torch.rand(8, 4).round()
        )
        child = mapper(population)
        assert child['x'].shape == torch.Size([4, 4])
    
    def test_binary_mapper_generates_population_after_two(self):

        mapper = tansaku.BinaryMapper(
            4
        )
        population = tansaku.Population(
            x=torch.rand(8, 4).round()
        )
        population2 = tansaku.Population(
            x=torch.rand(8, 4).round()
        )
        child = mapper(population)
        child = mapper(population2)
        assert child['x'].shape == torch.Size([4, 4])

    def test_binary_mapper_generates_population_after_two_with_signed_neg(self):

        mapper = tansaku.BinaryMapper(
            4, sign_neg=True
        )
        population = tansaku.Population(
            x=torch.randn(8, 4).sign()
        )
        population2 = tansaku.Population(
            x=torch.randn(8, 4).sign()
        )
        child = mapper(population)
        child = mapper(population2)
        assert child['x'].shape == torch.Size([4, 4])
        assert ((child['x'] == 1.0) | (child['x'] == -1.0)).all()


