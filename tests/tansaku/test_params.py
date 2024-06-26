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
