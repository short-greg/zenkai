import torch
from zenkai.tansaku import _reshape
import pytest


class TestExpandDim0:
    def test_expand_dim0_returns_correct_size_without_reshape(self):

        x = torch.randn(2, 4)
        x = _reshape.expand_dim0(x, 3)
        assert x.shape[0] == 3

    def test_expand_dim0_returns_correct_values_without_reshape(self):

        x = torch.randn(2, 4)
        y = _reshape.expand_dim0(x, 3)
        assert (x[None] == y).all()

    def test_expand_dim0_returns_correct_size_with_reshape(self):

        x = torch.randn(2, 4)
        x = _reshape.expand_dim0(x, 3, reshape=True)
        assert x.shape[0] == 6

    def test_expand_dim0_raises_error_with_incorrect_k(self):

        x = torch.randn(2, 4)
        with pytest.raises(ValueError):
            _reshape.expand_dim0(x, -1, reshape=True)



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
        shape = _reshape.collapse_batch(x_trial).shape
        assert shape[0] == x_trial.shape[0] * x_trial.shape[1]


class TestExpandK:
    def test_collapse_k_collapses_the_trial_dimension(
        self, x_trial_collapsed: torch.Tensor
    ):
        shape = _reshape.separate_batch(x_trial_collapsed, N_TRIALS).shape
        assert shape[0] == N_TRIALS
        assert shape[1] == N_SAMPLES



class TestFlattenDim0:
    def test_flatten_dim0_combines_first_two_dimensions(self):

        x = torch.randn(2, 4, 2)
        x = _reshape.flatten_dim0(x)
        assert x.shape[0] == 8

    def test_flatten_dim0_leaves_tensor_same_if_one_dimensional(self):

        x = torch.randn(2)
        x = _reshape.flatten_dim0(x)
        assert x.shape[0] == 2


class TestDeflattenDim0:
    def test_deflatten_dim0_undoes_the_flattening(self):

        x = torch.randn(8, 2)
        x = _reshape.deflatten_dim0(x, 2)
        assert x.shape[0] == 2
        assert x.shape[1] == 4