# 3rd party
import torch

# local
from zenkai.nnz._hard import Argmax, Sign


class TestArgmax:
    def test_argmax_returns_indices_of_max_along_dim(self):
        x = torch.tensor([[1, 3, 2], [4, 0, 5]])
        result = Argmax(dim=1)(x)
        assert torch.equal(result, torch.tensor([1, 2]))

    def test_argmax_default_dim_is_last(self):
        x = torch.tensor([[1, 3, 2], [4, 0, 5]])
        assert torch.equal(Argmax()(x), Argmax(dim=-1)(x))

    def test_argmax_keepdim_retains_reduced_dim(self):
        x = torch.tensor([[1, 3, 2], [4, 0, 5]])
        result = Argmax(dim=1, keepdim=True)(x)
        assert result.shape == (2, 1)


class TestSign:
    def test_sign_returns_sign_of_each_element(self):
        x = torch.tensor([[-1.5, 0.0, 2.3], [4.1, -0.2, -3.3]])
        result = Sign()(x)
        expected = torch.tensor([[-1.0, 0.0, 1.0], [1.0, -1.0, -1.0]])
        assert torch.equal(result, expected)

    def test_sign_preserves_shape(self):
        x = torch.randn(3, 4)
        assert Sign()(x).shape == x.shape
