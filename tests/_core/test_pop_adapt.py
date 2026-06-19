import torch
import torch.nn as nn

from zenkai._core import _pop_adapt as pop_adapt


class TestFeatureAdapt:
    def test_feature_adapt_preserves_population_shape(self):
        pop, batch, features = 3, 4, 5
        # feature_adapt folds the population into the feature dim, so the
        # module sees pop * features inputs.
        module = nn.Linear(pop * features, pop * features)
        x = torch.randn(pop, batch, features)
        y = pop_adapt.feature_adapt(module, x, dim=2)
        assert y.shape == torch.Size([pop, batch, features])

    def test_feature_adapt_matches_direct_application_for_identity(self):
        pop, batch, features = 2, 3, 4
        module = nn.Identity()
        x = torch.randn(pop, batch, features)
        y = pop_adapt.feature_adapt(module, x, dim=2)
        assert torch.allclose(y, x)

    def test_feature_adapt_returns_tuple_for_tuple_output(self):
        pop, batch, features = 2, 3, 4

        class TwoOut(nn.Module):
            def forward(self, x):
                return x, x + 1

        x = torch.randn(pop, batch, features)
        y = pop_adapt.feature_adapt(TwoOut(), x, dim=2)
        assert isinstance(y, tuple)
        assert y[0].shape == torch.Size([pop, batch, features])
        assert y[1].shape == torch.Size([pop, batch, features])


class TestBatchAdapt:
    def test_batch_adapt_preserves_population_shape(self):
        pop, batch, features = 3, 4, 5
        module = nn.Linear(features, 2)
        x = torch.randn(pop, batch, features)
        y = pop_adapt.batch_adapt(module, x)
        assert y.shape == torch.Size([pop, batch, 2])

    def test_batch_adapt_matches_direct_application(self):
        pop, batch, features = 2, 3, 4
        module = nn.Linear(features, features)
        x = torch.randn(pop, batch, features)
        y = pop_adapt.batch_adapt(module, x)
        expected = module(x)
        assert torch.allclose(y, expected)

    def test_batch_adapt_returns_tuple_for_tuple_output(self):
        pop, batch, features = 2, 3, 4

        class TwoOut(nn.Module):
            def forward(self, x):
                return x, x + 1

        x = torch.randn(pop, batch, features)
        y = pop_adapt.batch_adapt(TwoOut(), x)
        assert isinstance(y, tuple)
        assert y[0].shape == torch.Size([pop, batch, features])
        assert y[1].shape == torch.Size([pop, batch, features])
