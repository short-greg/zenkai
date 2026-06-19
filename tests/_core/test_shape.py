import pytest
import torch

from zenkai._core import _shape


class TestSeparate:

    def test_separate_feature_with_reshape(self):

        x = torch.randn(4, 2)
        x2 = _shape.feature_separate(x, 2, reshape=True)
        assert x2.shape == torch.Size([2, 4, 1])

    def test_separate_feature_with_feature_dim3(self):

        x = torch.randn(4, 2, 4)
        x2 = _shape.feature_separate(x, 2, 3, reshape=True)
        assert x2.shape == torch.Size([2, 4, 2, 2])

    def test_separate_feature_with_view(self):

        x = torch.randn(4, 2)
        x2 = _shape.feature_separate(x, 2, reshape=False)
        assert x2.shape == torch.Size([2, 4, 1])

    def test_separate_batch_with_reshape(self):

        x = torch.randn(4, 2)
        x2 = _shape.batch_separate(x, 2, reshape=True)
        assert x2.shape == torch.Size([2, 2, 2])

    def test_separate_batch_with_view(self):

        x = torch.randn(4, 2)
        x2 = _shape.batch_separate(x, 2, reshape=False)
        assert x2.shape == torch.Size([2, 2, 2])


class TestCollapse:

    def test_collapse_feature_with_reshape(self):

        x = torch.randn(2, 4, 1)
        x2 = _shape.feature_collapse(x, reshape=True)
        assert x2.shape == torch.Size([4, 2])

    def test_collapse_feature_with_view(self):

        x = torch.randn(2, 4, 1)
        x2 = _shape.feature_collapse(x, reshape=False)
        assert x2.shape == torch.Size([4, 2])

    def test_collapse_feature_with_feature_dim3(self):

        x = torch.randn(2, 4, 1, 3)
        x2 = _shape.feature_collapse(x, 3, reshape=True)
        assert x2.shape == torch.Size([4, 1, 6])

    def test_collapse_batch_with_reshape(self):

        x = torch.randn(2, 2, 2)
        x2 = _shape.batch_collapse(x, reshape=True)
        assert x2.shape == torch.Size([4, 2])

    def test_collapse_batch_with_view(self):

        x = torch.randn(2, 2, 2)
        x2 = _shape.batch_collapse(x, reshape=False)
        assert x2.shape == torch.Size([4, 2])


N_TRIALS = 4

N_SAMPLES = 3


def g(seed: int):
    g = torch.Generator()
    g.manual_seed(seed)


@pytest.fixture
def x():
    return torch.rand(N_SAMPLES, 2, generator=g(2))


@pytest.fixture
def x_trial():
    return torch.rand(N_TRIALS, N_SAMPLES, 2, generator=g(2))


@pytest.fixture
def x_trial_collapsed():
    return torch.rand(N_TRIALS * N_SAMPLES, 2, generator=g(2))


class TestCollapseK:
    def test_collapse_k_collapses_the_trial_dimension(self, x_trial: torch.Tensor):
        shape = _shape.batch_collapse(x_trial).shape
        assert shape[0] == x_trial.shape[0] * x_trial.shape[1]


class TestExpandK:
    def test_collapse_k_collapses_the_trial_dimension(self, x_trial_collapsed: torch.Tensor):
        shape = _shape.batch_separate(x_trial_collapsed, N_TRIALS).shape
        assert shape[0] == N_TRIALS
        assert shape[1] == N_SAMPLES
