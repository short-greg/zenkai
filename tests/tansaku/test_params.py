import torch
import torch.nn as nn
from zenkai.tansaku import _params as p_utils

# from zenkai.tansaku._params import set_p, acc_g
# from zenkai.tansaku import BestSelector, pop_assess, add_pop_noise
# from zenkai.utils import get_model_params, get_model_grads


class PopLinear(nn.Module):

    def __init__(self, pop_size: int, in_features: int, out_features: int):

        super().__init__()
        self.w = nn.parameter.Parameter(
            torch.randn(pop_size, in_features, out_features) * 0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x @ self.w


class TestPVec(object):

    def test_to_pvec_gets_correct_number(self):

        linear = PopLinear(4, 2, 2)
        pvec = p_utils.to_pvec(
            linear, 4
        )
        assert pvec.shape == torch.Size([4, 4])

    def test_to_pvec_and_back_is_same_value(self):

        linear = PopLinear(4, 2, 2)
        linear2 = PopLinear(4, 2, 2)
        
        pvec = p_utils.to_pvec(
            linear, 4
        )
        p_utils.set_pvec(
            linear2, pvec
        )
        
        assert (linear.w == linear2.w).all()

    def test_acc_pvec_is_two_times_original(self):

        linear = PopLinear(4, 2, 2)
        linear2 = PopLinear(4, 2, 2)
        pvec = p_utils.to_pvec(
            linear, 4
        )
        p_utils.set_pvec(
            linear2, pvec
        )
        p_utils.acc_pvec(
            linear2, pvec
        )
        assert ((linear.w * 2) == linear2.w).all()

    def test_acc_grad_produces_same_vals_as_weights(self):

        linear = PopLinear(4, 2, 2)
        
        pvec = p_utils.to_pvec(
            linear, 4
        )
        p_utils.acc_gradvec(
            linear, pvec
        )
        
        assert (linear.w == linear.w.grad).all()

    def test_acc_grad_produces_two_times_vals_as_weights(self):

        linear = PopLinear(4, 2, 2)
        
        pvec = p_utils.to_pvec(
            linear, 4
        )
        p_utils.acc_gradvec(
            linear, pvec
        )
        
        p_utils.acc_gradvec(
            linear, pvec
        )
        assert ((linear.w * 2) == linear.w.grad).all()

    def test_set_grad_produces_same_val_as_original(self):

        linear = PopLinear(4, 2, 2)
        
        pvec = p_utils.to_pvec(
            linear, 4
        )
        p_utils.acc_gradvec(
            linear, pvec
        )
        
        p_utils.set_gradvec(
            linear, pvec
        )
        assert ((linear.w) == linear.w.grad).all()


# class TestPopParams(object):

#     def test_update_pop_params_works_for_pop_linear(self):

#         linear = PopLinear(4, 3, 8)
#         x = torch.randn(4, 3, 3)
#         t = torch.randn(1, 3, 8)
#         y = linear(x)
#         assessment = pop_assess((y - t).pow(2), 'mean', 1)
#         selector = BestSelector(0)
#         selection = selector(assessment)

#         def update(p, assessment):
#             return add_pop_noise(
#                 p, 4, lambda x, info: x + torch.randn(
#                     info.shape, device=info.device, dtype=info.dtype))


#         before = get_model_params(linear)
#         set_p(linear, selection, update)
#         assert (get_model_params(linear) != before).any()


# class TestPopGrad(object):

#     def test_update_pop_grads_works_for_pop_linear(self):

#         linear = PopLinear(4, 3, 8)
#         x = torch.randn(4, 3, 3)
#         t = torch.randn(1, 3, 8)
#         y = linear(x)
#         assessment = pop_assess((y - t).pow(2), 'mean', 1)
#         selector = BestSelector(0)
#         selection = selector(assessment)

#         def update(p, assessment):
#             return add_pop_noise(
#                 p, 4, lambda x, info: x + torch.randn(
#                     info.shape, device=info.device, dtype=info.dtype))


#         before = get_model_grads(linear)
#         acc_g(linear, selection, update)
#         assert (get_model_grads(linear) != before)

#     def test_update_pop_grads_works_for_pop_linear_with_grads_set(self):

#         linear = PopLinear(4, 3, 8)
#         linear.w.grad = torch.randn_like(linear.w)
#         x = torch.randn(4, 3, 3)
#         t = torch.randn(1, 3, 8)
#         y = linear(x)
#         assessment = pop_assess((y - t).pow(2), 'mean', 1)
#         selector = BestSelector(0)
#         selection = selector(assessment)

#         def update(p, assessment):
#             return add_pop_noise(
#                 p, 4, lambda x, info: x + torch.randn(
#                     info.shape, device=info.device, dtype=info.dtype))


#         before = get_model_grads(linear, flat_cat=True)
#         acc_g(linear, selection, update)
#         assert (get_model_grads(linear, flat_cat=True) != before).any()
