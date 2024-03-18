# 3rd party
import torch
import torch.nn as nn
import numpy as np

import pytest

from zenkai.utils import _convert


class TestToTH:
    def test_to_th_converts_to_correct_dtype(self):

        x = np.random.randn(2, 4)
        y = _convert.to_th(x, dtype=torch.float64)
        assert y.dtype == torch.float64

    def test_to_th_sets_requires_grad(self):

        x = np.random.randn(2, 4)
        y = _convert.to_th(x, dtype=torch.float64, requires_grad=True)
        assert y.requires_grad is True

    def test_to_th_sets_retain_grad(self):

        x = np.random.randn(2, 4)
        y = _convert.to_th(
            x, dtype=torch.float64, requires_grad=True, retains_grad=True
        )
        (y).sum().backward()
        assert y.grad is not None


class TestToTHAs:
    def test_to_th_converts_to_correct_dtype(self):

        x = np.random.randn(2, 4)
        y = torch.randn(2, 4)
        x = _convert.to_th_as(x, y)
        assert x.dtype == y.dtype

    def test_to_th_sets_requires_grad(self):

        x = np.random.randn(2, 4)
        y = torch.randn(2, 4)
        x = _convert.to_th_as(x, y, requires_grad=True)
        assert x.requires_grad is True

    def test_to_th_sets_retain_grad(self):

        x = np.random.randn(2, 4)
        y = torch.randn(2, 4)
        x = _convert.to_th_as(x, y, requires_grad=True, retains_grad=True)
        (x).sum().backward()
        assert x.grad is not None


class TestExpandDim0:
    def test_expand_dim0_returns_correct_size_without_reshape(self):

        x = torch.randn(2, 4)
        x = _convert.expand_dim0(x, 3)
        assert x.shape[0] == 3

    def test_expand_dim0_returns_correct_values_without_reshape(self):

        x = torch.randn(2, 4)
        y = _convert.expand_dim0(x, 3)
        assert (x[None] == y).all()

    def test_expand_dim0_returns_correct_size_with_reshape(self):

        x = torch.randn(2, 4)
        x = _convert.expand_dim0(x, 3, reshape=True)
        assert x.shape[0] == 6

    def test_expand_dim0_raises_error_with_incorrect_k(self):

        x = torch.randn(2, 4)
        with pytest.raises(ValueError):
            _convert.expand_dim0(x, -1, reshape=True)


class TestFlattenDim0:
    def test_flatten_dim0_combines_first_two_dimensions(self):

        x = torch.randn(2, 4, 2)
        x = _convert.flatten_dim0(x)
        assert x.shape[0] == 8

    def test_flatten_dim0_leaves_tensor_same_if_one_dimensional(self):

        x = torch.randn(2)
        x = _convert.flatten_dim0(x)
        assert x.shape[0] == 2


class TestDeflattenDim0:
    def test_deflatten_dim0_undoes_the_flattening(self):

        x = torch.randn(8, 2)
        x = _convert.deflatten_dim0(x, 2)
        assert x.shape[0] == 2
        assert x.shape[1] == 4


class TestFreshen:
    def test_freshen_retains_grad_if_require_grad_is_true(self):

        x = torch.randn(8, 2)
        x = _convert.freshen(x, True)
        (x).sum().backward()
        assert x.grad is not None

    def test_freshen_detaches_grad_function(self):

        x_base = _convert.freshen(torch.rand(8, 2), True) * 2
        x = _convert.freshen(x_base)
        assert x.grad_fn is None

    def test_freshen_does_nothing_if_not_tensor(self):

        x = _convert.freshen(4, True)
        assert x == 4


class TestLR:

    def test_lr_update_interpolates_between_current_and_new(self):

        x = torch.rand(2, 4)
        y = torch.rand(2, 4)
        z = _convert.lr_update(x, y, 0.2)
        assert (z == (y * 0.2 + x * 0.8)).all()

    def test_lr_update_returns_new_if_no_learning_rate(self):

        x = torch.rand(2, 4)
        y = torch.rand(2, 4)
        z = _convert.lr_update(x, y)
        assert (z == y).all()


class TestBinaryEncoding(object):
    def test_binary_encoding_outputs_correct_size_with_twod(self):
        torch.manual_seed(1)

        x = (torch.rand(4) * 4).long()
        encoding = _convert.binary_encoding(x, 4)
        assert encoding.shape == torch.Size([4, 2])

    def test_binary_encoding_outputs_correct_size_with_threed(self):
        torch.manual_seed(1)

        x = (torch.rand(4, 2) * 4).long()
        encoding = _convert.binary_encoding(x, 4)
        assert encoding.shape == torch.Size([4, 2, 2])

    def test_binary_encoding_outputs_correct_size_with_bits_passed_in(self):
        torch.manual_seed(1)

        x = (torch.rand(4, 2) * 4).long()
        encoding = _convert.binary_encoding(x, 2, True)
        assert encoding.shape == torch.Size([4, 2, 2])


class TestToSignedNeg:
    def test_to_signed_neg_converts_zero_to_neg1(self):

        x = torch.zeros(2, 4)
        x = _convert.to_signed_neg(x)
        assert (x == -1).all()

    def test_to_signed_neg_leaves_one_as_one(self):

        x = torch.ones(2, 4)
        x = _convert.to_signed_neg(x)
        assert (x == 1).all()


class TestToZeroNeg:
    def test_to_zero_neg_converts_neg1_to_zero(self):

        x = -torch.ones(2, 4)
        x = _convert.to_zero_neg(x)
        assert (x == 0).all()

    def test_to_signed_neg_leaves_one_as_one(self):

        x = torch.ones(2, 4)
        x = _convert.to_zero_neg(x)
        assert (x == 1).all()


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
        shape = _convert.collapse_k(x_trial).shape
        assert shape[0] == x_trial.shape[0] * x_trial.shape[1]


class TestExpandK:
    def test_collapse_k_collapses_the_trial_dimension(
        self, x_trial_collapsed: torch.Tensor
    ):
        shape = _convert.expand_k(x_trial_collapsed, N_TRIALS).shape
        assert shape[0] == N_TRIALS
        assert shape[1] == N_SAMPLES
