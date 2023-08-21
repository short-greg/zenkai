import torch
import torch.nn as nn

from zenkai.utils import modules


class TestArgmax:

    def test_argmax_returns_last(self):

        argmax = modules.Argmax()
        x = torch.cumsum(torch.rand(2, 4), dim=-1)
        result = argmax(x)
        assert result[0] == 3

    def test_argmax_returns_first(self):

        argmax = modules.Argmax()
        x = torch.cumsum(torch.rand(2, 4), dim=-1).sort(-1, True)[0]
        result = argmax(x)
        assert result[0] == 0
