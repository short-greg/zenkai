# 3rd party
import pytest
import torch

# local
from zenkai._core import _noise as core_noise


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
        samples = core_noise.sample_gaussian(mean, std, 5)
        assert samples.shape == torch.Size([5, *mean.shape])

    def test_gaussian_sample_with_no_k(self):

        mean = torch.randn(4)
        std = torch.rand(4)
        samples = core_noise.noise_gaussian(mean, std)
        assert samples.shape == mean.shape


class TestNoise(object):

    def test_gaussian_noise(self):

        x = torch.randn(4)
        mean = torch.randn(4)
        std = torch.rand(4)
        samples = core_noise.noise_gaussian(x, mean, std)
        assert samples.shape == mean.shape

    def test_binary_noise(self):

        x = torch.rand(4).round()
        p = torch.randn(4)
        samples = core_noise.noise_binary(x, p, False)
        assert samples.shape == x.shape
        assert ((samples >= 0.0) | (samples <= 1.0)).all()

    def test_binary_noise_with_signed(self):

        x = torch.randn(4).sign()
        p = torch.randn(4)
        samples = core_noise.noise_binary(x, p, True)

        assert samples.shape == x.shape
        assert ((samples >= -1.0) | (samples <= 1.0)).all()
