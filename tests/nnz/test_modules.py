# 3rd party
import torch

# local
from zenkai.nnz._modules import Lambda, Null


class TestLambda:

    def test_lambda_applies_function(self):

        layer = Lambda(lambda x: x + 1)
        x = torch.randn(2, 2)
        y = layer(x)
        assert (y == x + 1).all()

    def test_lambda_passes_args_and_kwargs(self):

        layer = Lambda(lambda x, y, *, z: x + y + z, 2, z=3)
        x = torch.randn(2, 2)
        y = layer(x)
        assert torch.allclose(y, x + 5)


class TestNull:

    def test_null_forward(self):

        null = Null()
        x = torch.randn(2, 2)
        y = null(x)
        assert x is y

    def test_null_forward_with_multi(self):

        null = Null()
        x = torch.randn(2, 2)
        x1 = torch.randn(2, 2)
        y, y1 = null(x, x1)
        assert x is y
        assert x1 is y1

    def test_null_reverse(self):

        null = Null()
        x = torch.randn(2, 2)
        y = null.reverse(x)
        assert x is y

    def test_null_reverse_with_multi(self):

        null = Null()
        x = torch.randn(2, 2)
        x1 = torch.randn(2, 2)
        y, y1 = null.reverse(x, x1)
        assert x is y
        assert x1 is y1
