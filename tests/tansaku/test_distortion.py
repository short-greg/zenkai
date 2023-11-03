

# 3rd party
import torch
import torch.nn as nn

# local
from zenkai import tansaku
from zenkai import kaku


class TestGaussianNoiser:

    def test_gaussian_mapper_generates_gaussian_population_after_one(self):

        mapper = tansaku.GaussianNoiser()
        population = kaku.Population(
            x=torch.randn(8, 4)
        )
        child = mapper(population)
        assert child['x'].shape == torch.Size([8, 4])
    
    def test_gaussian_mapper_generates_gaussian_population_after_one_with_std_and_mu(self):

        mapper = tansaku.GaussianNoiser(
            mean=0.0, std=1.0
        )
        population = kaku.Population(
            x=torch.randn(8, 4)
        )
        child = mapper(population)
        assert child['x'].shape == torch.Size([8, 4])

    def test_gaussian_mapper_generates_gaussian_population_after_two_updates(self):

        mapper = tansaku.GaussianNoiser()
        population = kaku.Population(
            x=torch.randn(8, 4)
        )
        population2 = kaku.Population(
            x=torch.randn(8, 4)
        )
        child = mapper(population)
        child = mapper(population2)
        assert child['x'].shape == torch.Size([8, 4])


class TestBinarySampler:

    def test_binary_mapper_generates_population_after_one(self):

        mapper = tansaku.BinaryNoiser()
        population = kaku.Population(
            x=torch.rand(8, 4).round()
        )
        child = mapper(population)
        assert child['x'].shape == torch.Size([8, 4])
    
    def test_binary_mapper_generates_population_after_two(self):

        mapper = tansaku.BinaryNoiser(0.3)
        population = kaku.Population(
            x=torch.rand(8, 4).round()
        )
        population2 = kaku.Population(
            x=torch.rand(8, 4).round()
        )
        child = mapper(population)
        child = mapper(population2)
        assert child['x'].shape == torch.Size([8, 4])

    def test_binary_mapper_generates_population_after_two_with_signed_neg(self):

        mapper = tansaku.BinaryNoiser(
            signed_neg=True
        )
        population = kaku.Population(
            x=torch.randn(8, 4).sign()
        )
        population2 = kaku.Population(
            x=torch.randn(8, 4).sign()
        )
        child = mapper(population)
        child = mapper(population2)
        assert child['x'].shape == torch.Size([8, 4])
        assert ((child['x'] == 1.0) | (child['x'] == -1.0)).all()
