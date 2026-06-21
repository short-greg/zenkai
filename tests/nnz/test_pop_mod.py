import torch
import torch.nn as nn

from zenkai._core import crossover_full
from zenkai.nnz._pop_mod import (
    AdaptPopBatch,
    AdaptPopFeature,
    CrossOver,
    NullPopAdapt,
)


class TestCrossOver:

    def test_crossover_generates_child(self):

        x1 = torch.rand(3, 2)
        x2 = torch.rand(3, 2)
        crossover = CrossOver(crossover_full, p1=0.5)
        child = crossover(x1, x2)
        assert child.shape == torch.Size([3, 2])

    def test_crossover_without_f_returns_first_parent(self):

        x1 = torch.rand(3, 2)
        x2 = torch.rand(3, 2)
        crossover = CrossOver()
        child = crossover(x1, x2)
        assert (child == x1).all()


class TestAdaptPopBatch:

    def test_adapt_pop_batch_collapses_and_separates_population(self):

        module = nn.Linear(4, 3)
        adapt = AdaptPopBatch(module)
        # population of 2, batch of 5, features 4
        x = torch.randn(2, 5, 4)
        y = adapt((x,))
        assert y.shape == torch.Size([2, 5, 3])


class TestAdaptPopFeature:

    def test_adapt_pop_feature_collapses_and_separates_feature(self):

        module = nn.Linear(4, 3)
        adapt = AdaptPopFeature(module, n_members=2, feature_dim=1)
        # batch of 5, population of 2, features 4
        x = torch.randn(5, 2, 4)
        y = adapt(x)
        # feature_collapse merges dims 0/1 -> (10, 4); the module applies and
        # separates back using k = x[0].size(0) == 2 -> (10, 1, 3)
        assert y.shape == torch.Size([10, 1, 3])


class TestNullPopAdapt:

    def test_null_pop_adapt_passes_through(self):

        module = nn.Linear(4, 3)
        adapt = NullPopAdapt(module, n_members=2)
        x = torch.randn(2, 5, 4)
        y = adapt(x)
        assert y.shape == torch.Size([2, 5, 3])
        assert torch.allclose(y, module(x))
