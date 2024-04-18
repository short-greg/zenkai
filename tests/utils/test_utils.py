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
