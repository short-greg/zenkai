from torch import Tensor
from zenkai.utils.params import _pop_params
from typing_extensions import Self

import torch
import torch.nn as nn


class PopLinear(_pop_params.PopModule):
    
    def __init__(self, n_members: int = None):
        super().__init__(n_members)

        if n_members is None:
            self.w = nn.parameter.Parameter(
                torch.rand(4, 4)
            )
        else:
            self.w = nn.parameter.Parameter(
                torch.rand(n_members, 4, 4)
            )

    def spawn(self, n_members: int) -> Self:
        return PopLinear(n_members)
    
    def forward(self, x: Tensor) -> Tensor:
        return x @ self.w


class TestPopModule:

    def test_pop_linear_outputs_pop_size_of4(self):

        linear = PopLinear(4)
        x = torch.randn(1, 5, 4)
        assert (linear(x).shape[0] == 4)

    def test_pop_linear_outputs_pop_size_of_none(self):

        linear = PopLinear(None)
        x = torch.randn(5, 4)
        assert (linear(x).shape[0] == 5)

    def test_pop_linear_spawned_outputs_correct_size(self):

        linear = PopLinear(None)
        linear2 = linear.spawn(3)
        x = torch.randn(1, 5, 4)
        assert (linear2(x).shape[0] == 3)
