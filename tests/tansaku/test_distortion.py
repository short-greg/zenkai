# 3rd party
import torch

# local
from zenkai.tansaku import _distortion
from zenkai import kaku

from functools import partial

class TestGaussianNoiser:
    def test_gaussian_mapper_generates_gaussian_population_after_one(self):

        population = kaku.Population(x=torch.randn(8, 4))
        child = population.apply(partial(_distortion.gausian_noise, std=0.1))
        assert child["x"].shape == torch.Size([8, 4])

    def test_gaussian_mapper_generates_gaussian_population_after_one_with_std_and_mu(
        self,
    ):

        population = kaku.Population(x=torch.randn(8, 4))
        child = population.apply(partial(_distortion.gausian_noise, std=0.1, mean=0.1))
        assert child["x"].shape == torch.Size([8, 4])

    def test_gaussian_mapper_generates_gaussian_population_after_two_updates(self):

        population = kaku.Population(x=torch.randn(8, 4))
        population2 = kaku.Population(x=torch.randn(8, 4))
        child = population.apply(partial(_distortion.gausian_noise, std=0.1, mean=0.1))
        child = population2.apply(partial(_distortion.gausian_noise, std=0.1, mean=0.1))
        assert child["x"].shape == torch.Size([8, 4])


class TestBinarySampler:

    def test_binary_mapper_generates_population_after_one(self):

        population = kaku.Population(x=torch.rand(8, 4).round())
        child = population.apply(partial(_distortion.binary_noise, flip_p=0.5))
        assert child["x"].shape == torch.Size([8, 4])

    def test_binary_mapper_generates_population_after_two(self):

        population = kaku.Population(x=torch.rand(8, 4).round())
        population2 = kaku.Population(x=torch.rand(8, 4).round())
        child = population.apply(partial(_distortion.binary_noise, flip_p=0.5))
        child = population2.apply(partial(_distortion.binary_noise, flip_p=0.5))
        assert child["x"].shape == torch.Size([8, 4])

    def test_binary_mapper_generates_population_after_two_with_signed_neg(self):

        population = kaku.Population(x=torch.randn(8, 4).sign())
        population2 = kaku.Population(x=torch.randn(8, 4).sign())
        child = population.apply(partial(_distortion.binary_noise, flip_p=0.5))
        child = population2.apply(partial(_distortion.binary_noise, flip_p=0.5))
        assert child["x"].shape == torch.Size([8, 4])
        assert ((child["x"] == 1.0) | (child["x"] == -1.0)).all()
