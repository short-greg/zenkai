import torch

from zenkai.nnz._shape import ExpandDim


class TestExpandDim:
    def test_expand_dim_inserts_dimension(self):
        x = torch.tensor([1, 2, 3])
        expand = ExpandDim(dim=0, size1=3, size2=1)
        result = expand(x)
        assert result.shape == torch.Size([3, 1])

    def test_expand_dim_preserves_values(self):
        x = torch.tensor([1, 2, 3])
        expand = ExpandDim(dim=0, size1=3, size2=1)
        result = expand(x)
        assert torch.equal(result, torch.tensor([[1], [2], [3]]))
