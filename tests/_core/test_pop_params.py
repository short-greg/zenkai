import torch
import torch.nn as nn

from zenkai._core import _pop_params as p_utils


class PopLinear(p_utils.PopModule):

    def __init__(self, pop_size: int, in_features: int, out_features: int):

        super().__init__(pop_size)
        self.w = nn.parameter.Parameter(torch.randn(pop_size, in_features, out_features) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x @ self.w


class TestPVec(object):

    def test_to_pvec_gets_correct_number(self):

        linear = PopLinear(4, 2, 2)
        pvec = p_utils.to_pop_pvec(linear, 4)
        assert pvec.shape == torch.Size([4, 4])

    def test_to_pvec_and_back_is_same_value(self):

        linear = PopLinear(4, 2, 2)
        linear2 = PopLinear(4, 2, 2)

        pvec = p_utils.to_pop_pvec(linear, 4)
        p_utils.pop_vec_set(linear2, pvec)
        assert torch.isclose(linear.w, linear2.w).all()

    def test_acc_pvec_is_two_times_original(self):

        linear = PopLinear(4, 2, 2)
        linear2 = PopLinear(4, 2, 2)
        pvec = p_utils.to_pop_pvec(linear, 4)
        p_utils.pop_vec_set(linear2, pvec)
        p_utils.pop_vec_acc(linear2, pvec)
        assert ((linear.w * 2) == linear2.w).all()

    def test_acc_grad_produces_same_vals_as_weights(self):

        linear = PopLinear(4, 2, 2)

        pvec = p_utils.to_pop_pvec(linear, 4)
        p_utils.pop_gradvec_acc(linear, pvec)

        assert (linear.w == linear.w.grad).all()

    def test_acc_grad_produces_two_times_vals_as_weights(self):

        linear = PopLinear(4, 2, 2)

        pvec = p_utils.to_pop_pvec(linear, 4)
        p_utils.pop_gradvec_acc(linear, pvec)

        p_utils.pop_gradvec_acc(linear, pvec)
        assert ((linear.w * 2) == linear.w.grad).all()

    def test_acc_gradtvec_produces_two_times_vals_as_weights(self):
        linear = PopLinear(4, 2, 2)

        pvec = p_utils.to_pop_pvec(linear, 4)
        p_utils.pop_gradtvec_acc(linear, pvec)
        p_utils.pop_gradtvec_acc(linear, pvec)
        assert (torch.zeros_like(linear.w) == linear.w.grad).all()

    def test_set_grad_produces_same_val_as_original(self):

        linear = PopLinear(4, 2, 2)

        pvec = p_utils.to_pop_pvec(linear, 4)
        p_utils.pop_gradvec_acc(linear, pvec)

        p_utils.pop_gradvec_set(linear, pvec)
        assert ((linear.w) == linear.w.grad).all()
