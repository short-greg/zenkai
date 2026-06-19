# 3rd party
import torch
import torch.nn as nn

from zenkai._core import _params as p_utils


class TestSetModelParameters:

    def test_set_parameters_makes_the_two_modules_the_same(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)
        p_utils.model_params_update(mod2, p_utils.params_get(mod1))
        assert (p_utils.params_get(mod1) == p_utils.params_get(mod2)).all()


class TestGetP:

    def test_p_get_gets_all_ps_with_two_modules(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)

        res = [p for p in p_utils.p_get([mod1, mod2])]
        assert res[0] is mod1.weight or res[0] is mod1.bias
        assert res[2] is mod2.weight or res[2] is mod2.bias

    def test_p_get_gets_all_ps_with_parameters(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)

        res = [p for p in p_utils.p_get([mod1.parameters(), mod2.parameters()])]
        assert res[0] is mod1.weight or res[0] is mod1.bias
        assert res[2] is mod2.weight or res[2] is mod2.bias

    def test_p_get_gets_all_ps_with_callable(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)

        res = [p for p in p_utils.p_get([mod1.parameters, mod2.parameters()])]
        assert res[0] is mod1.weight or res[0] is mod1.bias
        assert res[2] is mod2.weight or res[2] is mod2.bias

    def test_p_get_gets_all_ps_with_tensor(self):

        x = torch.randn(2, 4, requires_grad=True)
        x2 = torch.randn(2, 4, requires_grad=True)

        p = [p for p in p_utils.p_get([x, x2])]
        assert p[0] is x
        assert p[1] is x2

    def test_p_get_gets_all_ps_with_single_tensor(self):

        x = torch.randn(2, 4, requires_grad=True)

        p = [p for p in p_utils.p_get(x)]
        assert p[0] is x


class TestPVec:

    def test_pvec_set_sets_back_to_original(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)

        vec = p_utils.to_pvec([mod1, mod2])
        weight_before = torch.clone(mod1.weight)
        p_utils.pvec_set([mod1, mod2], vec)
        assert (mod1.weight == weight_before).all()

    def test_set_p_vec_sets_to_new_vec_after_adding(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)

        vec = p_utils.to_pvec([mod1, mod2]) + 1.0
        weight_before = torch.clone(mod1.weight)
        p_utils.pvec_set([mod1, mod2], vec)
        assert (mod1.weight == (weight_before + 1.0)).all()

    def test_pvec_acc_doubles_the_pvec_value(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)

        vec = p_utils.to_pvec([mod1, mod2])
        weight_before = torch.clone(mod1.weight)
        p_utils.pvec_acc([mod1, mod2], vec)
        assert (mod1.weight == (weight_before * 2)).all()

    def test_gradvec_set_sets_all_grads_to_same_value(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)

        vec = p_utils.to_pvec([mod1, mod2])
        weight_before = torch.clone(mod1.weight)
        p_utils.gradvec_set([mod1, mod2], vec)
        assert (mod1.weight.grad == weight_before).all()

    def test_gradvec_acc_sets_all_grads_to_same_value(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)

        vec = p_utils.to_pvec([mod1, mod2])
        weight_before = torch.clone(mod1.weight)
        p_utils.gradvec_acc([mod1, mod2], vec)
        assert (mod1.weight.grad == weight_before).all()

    def test_gradvec_acc_sets_all_grads_to_double_after_executing_wtice(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)

        vec = p_utils.to_pvec([mod1, mod2])
        weight_before = torch.clone(mod1.weight)
        p_utils.gradvec_acc([mod1, mod2], vec)
        p_utils.gradvec_acc([mod1, mod2], vec)
        assert (mod1.weight.grad == (weight_before * 2)).all()


class TestApply:

    def test_apply_updates_the_value_of_p(self):

        model = nn.Linear(2, 3)
        p_utils.p_apply(model, lambda p: torch.ones_like(p))
        assert (model.weight == 1).all()

    def test_grad_apply_updates_the_value_of_p_grad(self):

        model = nn.Linear(2, 3)
        model.weight.grad = torch.zeros_like(model.weight)
        p_utils.grad_apply(model, lambda p, g: g + 1, True)
        assert (model.weight.grad == 1.0).all()

    def test_grad_apply_updates_the_value_of_p_grad_with_none(self):

        model = nn.Linear(2, 3)
        model.weight.grad = torch.zeros_like(model.weight)
        p_utils.grad_apply(model, lambda p, g: torch.ones_like(p), False)
        assert (model.weight.grad == 1.0).all()
