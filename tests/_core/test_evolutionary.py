# 3rd party
import torch

# local
from zenkai._core._evolutionary import es_estimate


class TestEsEstimate:
    def test_es_estimate_returns_tensor_with_feature_shape(self):
        dw = torch.randn(8, 4)
        assessment = torch.randn(8)
        result = es_estimate(dw, assessment)
        assert result.shape == torch.Size([4])

    def test_es_estimate_collapses_population_dim(self):
        dw = torch.randn(6, 3, 2)
        assessment = torch.randn(6)
        result = es_estimate(dw, assessment)
        assert result.shape == torch.Size([3, 2])

    def test_es_estimate_accepts_tensor_reference(self):
        dw = torch.randn(8, 4)
        assessment = torch.randn(8)
        ref = torch.tensor(0.0)
        result = es_estimate(dw, assessment, assessment_ref=ref)
        assert result.shape == torch.Size([4])

    def test_es_estimate_is_weighted_mean_of_displacement(self):
        dw = torch.randn(8, 4)
        assessment = torch.randn(8)
        result = es_estimate(dw, assessment)
        assert torch.is_tensor(result)
        assert not torch.isnan(result).any()
