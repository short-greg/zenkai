import torch
import torch.nn as nn


from zenkai.tansaku._params import set_p, acc_g
from zenkai.tansaku import BestSelector, pop_assess, add_pop_noise
from zenkai.utils import get_model_params, get_model_grads


class PopLinear(nn.Module):

    def __init__(self, pop_size: int, in_features: int, out_features: int):

        super().__init__()
        self.w = nn.parameter.Parameter(
            torch.randn(pop_size, in_features, out_features) * 0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x @ self.w


class TestPopParams(object):

    def test_update_pop_params_works_for_pop_linear(self):

        linear = PopLinear(4, 3, 8)
        x = torch.randn(4, 3, 3)
        t = torch.randn(1, 3, 8)
        y = linear(x)
        assessment = pop_assess((y - t).pow(2), 'mean', 1)
        selector = BestSelector(0)
        selection = selector(assessment)

        def update(p, assessment):
            return add_pop_noise(
                p, 4, lambda x, info: x + torch.randn(
                    info.shape, device=info.device, dtype=info.dtype))


        before = get_model_params(linear)
        set_p(linear, selection, update)
        assert (get_model_params(linear) != before).any()


class TestPopGrad(object):

    def test_update_pop_grads_works_for_pop_linear(self):

        linear = PopLinear(4, 3, 8)
        x = torch.randn(4, 3, 3)
        t = torch.randn(1, 3, 8)
        y = linear(x)
        assessment = pop_assess((y - t).pow(2), 'mean', 1)
        selector = BestSelector(0)
        selection = selector(assessment)

        def update(p, assessment):
            return add_pop_noise(
                p, 4, lambda x, info: x + torch.randn(
                    info.shape, device=info.device, dtype=info.dtype))


        before = get_model_grads(linear)
        acc_g(linear, selection, update)
        assert (get_model_grads(linear) != before)

    def test_update_pop_grads_works_for_pop_linear_with_grads_set(self):

        linear = PopLinear(4, 3, 8)
        linear.w.grad = torch.randn_like(linear.w)
        x = torch.randn(4, 3, 3)
        t = torch.randn(1, 3, 8)
        y = linear(x)
        assessment = pop_assess((y - t).pow(2), 'mean', 1)
        selector = BestSelector(0)
        selection = selector(assessment)

        def update(p, assessment):
            return add_pop_noise(
                p, 4, lambda x, info: x + torch.randn(
                    info.shape, device=info.device, dtype=info.dtype))


        before = get_model_grads(linear, flat_cat=True)
        acc_g(linear, selection, update)
        assert (get_model_grads(linear, flat_cat=True) != before).any()
