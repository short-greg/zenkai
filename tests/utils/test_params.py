# 3rd party
import torch
import torch.nn as nn

from zenkai.utils import _params as p_utils


class TestSetModelParameters:

    def test_set_parameters_makes_the_two_modules_the_same(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)
        p_utils.update_model_params(mod2, p_utils.get_params(mod1))
        assert (
            p_utils.get_params(mod1) == p_utils.get_params(mod2)
        ).all()


class TestGetP:

    def test_get_p_gets_all_ps_with_two_modules(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)
        
        res = [p for p in p_utils.get_p([mod1, mod2])]
        assert res[0] is mod1.weight or res[0] is mod1.bias
        assert res[2] is mod2.weight or res[2] is mod2.bias

    def test_get_p_gets_all_ps_with_parameters(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)
        
        res = [p for p in p_utils.get_p([mod1.parameters(), mod2.parameters()])]
        assert res[0] is mod1.weight or res[0] is mod1.bias
        assert res[2] is mod2.weight or res[2] is mod2.bias

    def test_get_p_gets_all_ps_with_callable(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)
        
        res = [p for p in p_utils.get_p([mod1.parameters, mod2.parameters()])]
        assert res[0] is mod1.weight or res[0] is mod1.bias
        assert res[2] is mod2.weight or res[2] is mod2.bias

    def test_get_p_gets_all_ps_with_tensor(self):

        x = torch.randn(2, 4, requires_grad=True)
        x2 = torch.randn(2, 4, requires_grad=True)

        p = [p for p in p_utils.get_p([x, x2])]
        assert p[0] is x
        assert p[1] is x2

    def test_get_p_gets_all_ps_with_single_tensor(self):

        x = torch.randn(2, 4, requires_grad=True)

        p = [p for p in p_utils.get_p(x)]
        assert p[0] is x



class TestPVec:

    def test_set_pvec_sets_back_to_original(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)
        
        vec = p_utils.to_pvec([mod1, mod2])
        weight_before = torch.clone(mod1.weight)
        p_utils.set_pvec([mod1, mod2], vec)
        assert (mod1.weight == weight_before).all()

    def test_set_p_vec_sets_to_new_vec_after_adding(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)
        
        vec = p_utils.to_pvec([mod1, mod2]) + 1.0
        weight_before = torch.clone(mod1.weight)
        p_utils.set_pvec([mod1, mod2], vec)
        assert (mod1.weight == (weight_before + 1.0)).all()

    def test_acc_pvec_doubles_the_pvec_value(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)
        
        vec = p_utils.to_pvec([mod1, mod2])
        weight_before = torch.clone(mod1.weight)
        p_utils.acc_pvec([mod1, mod2], vec)
        assert (mod1.weight == (weight_before * 2)).all()

    def test_set_gradvec_sets_all_grads_to_same_value(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)
        
        vec = p_utils.to_pvec([mod1, mod2])
        weight_before = torch.clone(mod1.weight)
        p_utils.set_gradvec([mod1, mod2], vec)
        assert (mod1.weight.grad == weight_before).all()

    def test_acc_gradvec_sets_all_grads_to_same_value(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)
        
        vec = p_utils.to_pvec([mod1, mod2])
        weight_before = torch.clone(mod1.weight)
        p_utils.acc_gradvec([mod1, mod2], vec)
        assert (mod1.weight.grad == weight_before).all()

    def test_acc_gradvec_sets_all_grads_to_double_after_executing_wtice(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)
        
        vec = p_utils.to_pvec([mod1, mod2])
        weight_before = torch.clone(mod1.weight)
        p_utils.acc_gradvec([mod1, mod2], vec)
        p_utils.acc_gradvec([mod1, mod2], vec)
        assert (mod1.weight.grad == (weight_before * 2)).all()


class TestApply:

    def test_apply_updates_the_value_of_p(self):

        model = nn.Linear(2, 3)
        p_utils.apply_p(
            model, lambda p: torch.ones_like(p)
        )
        assert (model.weight == 1).all()


    def test_apply_grad_updates_the_value_of_p_grad(self):

        model = nn.Linear(2, 3)
        model.weight.grad = torch.zeros_like(
            model.weight
        )
        p_utils.apply_grad(
            model, lambda p, g: g + 1, True
        )
        assert (model.weight.grad == 1.).all()

    def test_apply_grad_updates_the_value_of_p_grad_with_none(self):

        model = nn.Linear(2, 3)
        model.weight.grad = torch.zeros_like(
            model.weight
        )
        p_utils.apply_grad(
            model, lambda p, g: torch.ones_like(p), False
        )
        assert (model.weight.grad == 1.).all()
