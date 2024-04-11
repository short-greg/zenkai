# 3rd party
import torch
import torch.nn as nn


from zenkai.utils import _params as p_utils
from zenkai.utils import undo_cat1d, cat1d


class TestSetModelParameters:
    def test_set_parameters_makes_the_two_modules_the_same(self):

        mod1 = nn.Linear(2, 4)
        mod2 = nn.Linear(2, 4)
        p_utils.update_model_params(mod2, p_utils.get_model_params(mod1))
        assert (
            p_utils.get_model_params(mod1) == p_utils.get_model_params(mod2)
        ).all()


class TestSetModelGrads:

    def test_set_model_grads_sets_correctly(self):

        mod1 = nn.Linear(2, 4)
        grads = torch.randn(12)

        p_utils.update_model_grads(mod1, undo_cat1d(mod1, grads), False)
        new_grads = p_utils.get_model_grads(mod1)

        assert (cat1d(new_grads) == grads).all()

    def test_set_model_grads_adds_grads_when_set_to_true(self):

        mod1 = nn.Linear(2, 4)
        grads1 = torch.randn(12)
        grads2 = torch.randn(12)

        p_utils.update_model_grads(mod1, undo_cat1d(mod1, grads1), False)
        p_utils.update_model_grads(mod1, undo_cat1d(mod1, grads2), True)
        new_grads = p_utils.get_model_grads(mod1)
        assert (cat1d(new_grads) == (grads1 + grads2)).all()

    def test_set_model_grads_does_not_add_when_not_set_to_true(self):

        mod1 = nn.Linear(2, 4)
        grads1 = torch.randn(12)
        grads2 = torch.randn(12)

        p_utils.update_model_grads(mod1, undo_cat1d(mod1, grads1), False)
        p_utils.update_model_grads(mod1, undo_cat1d(mod1, grads2), False)
        new_grads = p_utils.get_model_grads(mod1)
        assert (cat1d(new_grads) == grads2).all()

    def test_undo_grads_resets_grads(self):

        mod1 = nn.Linear(2, 4)
        grads1 = torch.randn(12)
        p_utils.update_model_grads(mod1, undo_cat1d(mod1, grads1), False)

        with p_utils.undo_grad([mod1]):
            grads2 = torch.randn(12)
            p_utils.update_model_grads(mod1, undo_cat1d(mod1, grads2), False)

        new_grads = p_utils.get_model_grads(mod1)
        assert (cat1d(new_grads) == grads1).all()

    def test_undo_grads_resets_grads_for_tensor(self):

        mod1 = nn.Linear(2, 4)
        grads1 = torch.randn(12)
        t = torch.rand(12)
        t_grad = torch.rand(12)
        t.grad = t_grad

        p_utils.update_model_grads(mod1, undo_cat1d(mod1, grads1), False)

        with p_utils.undo_grad([mod1, t]):
            grads2 = torch.randn(12)
            p_utils.update_model_grads(mod1, undo_cat1d(mod1, grads2), False)
            t.grad = torch.rand(12)

        new_grads = p_utils.get_model_grads(mod1)
        assert (cat1d(new_grads) == grads1).all()
        assert (t_grad == t.grad).all()

