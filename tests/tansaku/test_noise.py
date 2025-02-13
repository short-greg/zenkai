# 3rd party
import torch

# local
import pytest
import torch
import torch.nn as nn

# from zenkai import Assessment
from zenkai.tansaku._noise import (
    EqualsAssessmentDist,
    add_noise,
    add_pop_noise,
    cat_noise,
    cat_pop_noise
)
from zenkai.tansaku import _noise as tansaku_noise

def g(seed: int):
    g = torch.Generator()
    g.manual_seed(seed)


N_TRIALS = 4

N_SAMPLES = 3


@pytest.fixture
def x():
    return torch.rand(N_SAMPLES, 2, generator=g(2))


@pytest.fixture
def x_trial():
    return torch.rand(N_TRIALS, N_SAMPLES, 2, generator=g(2))


@pytest.fixture
def x_trial_collapsed():
    return torch.rand(N_TRIALS * N_SAMPLES, 2, generator=g(2))


@pytest.fixture
def noise():
    return torch.rand(N_SAMPLES, 2, generator=g(3))


class TestGaussianSample(object):

    def test_gaussian_sample(self):

        mean = torch.randn(4)
        std = torch.rand(4)
        samples = tansaku_noise.gaussian_sample(
            mean, std, 5
        )
        assert samples.shape == torch.Size([5, *mean.shape])

    def test_gaussian_sample_with_no_k(self):

        mean = torch.randn(4)
        std = torch.rand(4)
        samples = tansaku_noise.gaussian_noise(
            mean, std
        )
        assert samples.shape == mean.shape


class TestNoise(object):

    def test_gaussian_noise(self):

        x = torch.randn(4)
        mean = torch.randn(4)
        std = torch.rand(4)
        samples = tansaku_noise.gaussian_noise(
            x, mean, std
        )
        assert samples.shape == mean.shape

    def test_binary_noise(self):

        x = torch.rand(4).round()
        p = torch.randn(4)
        samples = tansaku_noise.binary_noise(
            x, p, False
        )
        assert samples.shape == x.shape
        assert ((samples >= 0.0) | (samples <= 1.0)).all()

    def test_binary_noise_with_signed(self):

        x = torch.randn(4).sign()
        p = torch.randn(4)
        samples = tansaku_noise.binary_noise(
            x, p, True
        )

        assert samples.shape == x.shape
        assert ((samples >= -1.0) | (samples <= 1.0)).all()


class TestAddNoise(object):

    def test_add_noise_updates_tensor(self):

        x = torch.randn(4, 3)

        torch.manual_seed(1)
        base_noise = torch.randn(2, 4, 3)
        torch.manual_seed(1)

        noisy = add_noise(
            x, 2, lambda x, info: x + torch.randn(info.shape, **info.attr)
        )
        assert (noisy == (base_noise + x)).all()


class TestCatNoise(object):

    def test_cat_noise_adds_to_tensor(self):

        x = torch.randn(4, 3)

        torch.manual_seed(1)
        base_noise = torch.randn(2, 4, 3)
        torch.manual_seed(1)

        noisy = cat_noise(
            x, 2, lambda x, info: x + base_noise
        )
        assert (noisy == torch.cat([x.unsqueeze(0), (base_noise + x)])).all()


class TestAddPopNoise(object):

    def test_cat_noise_adds_to_tensor(self):

        x = torch.randn(2, 4, 3)

        torch.manual_seed(1)
        base_noise = torch.randn(2, 2, 4, 3)
        t = (x.unsqueeze(1) + base_noise).reshape(
            4, 4, 3
        )
        torch.manual_seed(1)

        noisy = add_pop_noise(
            x, 2, lambda x, info: x + torch.randn(info.shape, **info.attr)
        )
        
        assert torch.isclose(noisy, t, atol=1e-4).all()


class TestCatPopNoise(object):

    def test_cat_noise_adds_to_tensor(self):

        x = torch.randn(2, 4, 3)

        torch.manual_seed(1)
        base_noise = torch.randn(2, 2, 4, 3)
        t = torch.cat([x, (x.unsqueeze(1) + base_noise).reshape(
            4, 4, 3
        )])
        torch.manual_seed(1)

        noisy = cat_pop_noise(
            x, 2, lambda x, info: x + torch.randn(info.shape, **info.attr)
        )
        
        assert torch.isclose(noisy, t, atol=1e-4).all()


# class TestGaussianNoiser:
#     def test_explore_gaussian_returns_correct_size(self, x: torch.Tensor):
#         explorer = GaussianNoiser()
#         noise = explorer(x)
#         assert noise.size() == x.size()

#     def test_raises_value_error_when_std_less_than_zero(self, x: torch.Tensor):
#         with pytest.raises(ValueError):
#             GaussianNoiser(-1)


# class TestGaussianNoiser:

#     pass

    # def test_gaussian_mapper_generates_gaussian_population_after_one(self):

    #     population = torch.randn(8, 4)
    #     child = population.apply(partial(_noise.gausian_noise, std=0.1))
    #     assert child["x"].shape == torch.Size([8, 4])

    # def test_gaussian_mapper_generates_gaussian_population_after_one_with_std_and_mu(
    #     self,
    # ):

    #     population = kaku.Population(x=torch.randn(8, 4))
    #     child = population.apply(partial(_noise.gausian_noise, std=0.1, mean=0.1))
    #     assert child["x"].shape == torch.Size([8, 4])

    # def test_gaussian_mapper_generates_gaussian_population_after_two_updates(self):

    #     population = kaku.Population(x=torch.randn(8, 4))
    #     population2 = kaku.Population(x=torch.randn(8, 4))
    #     child = population.apply(partial(_noise.gausian_noise, std=0.1, mean=0.1))
    #     child = population2.apply(partial(_noise.gausian_noise, std=0.1, mean=0.1))
    #     assert child["x"].shape == torch.Size([8, 4])


class TestBinarySampler:

    pass
    # def test_binary_mapper_generates_population_after_one(self):

    #     population = kaku.Population(x=torch.rand(8, 4).round())
    #     child = population.apply(partial(_noise.binary_noise, flip_p=0.5))
    #     assert child["x"].shape == torch.Size([8, 4])

    # def test_binary_mapper_generates_population_after_two(self):

    #     population = kaku.Population(x=torch.rand(8, 4).round())
    #     population2 = kaku.Population(x=torch.rand(8, 4).round())
    #     child = population.apply(partial(_noise.binary_noise, flip_p=0.5))
    #     child = population2.apply(partial(_noise.binary_noise, flip_p=0.5))
    #     assert child["x"].shape == torch.Size([8, 4])

    # def test_binary_mapper_generates_population_after_two_with_signed_neg(self):

    #     population = kaku.Population(x=torch.randn(8, 4).sign())
    #     population2 = kaku.Population(x=torch.randn(8, 4).sign())
    #     child = population.apply(partial(_noise.binary_noise, flip_p=0.5))
    #     child = population2.apply(partial(_noise.binary_noise, flip_p=0.5))
    #     assert child["x"].shape == torch.Size([8, 4])
    #     assert ((child["x"] == 1.0) | (child["x"] == -1.0)).all()

