# 3rd Party
import torch
from torch import nn
from torch import optim

# Local
from zenkai.kaku._optimize import ParamFilter, OptimFactory, NullOptim, optimf
from zenkai.utils import get_model_params


class TestParamFilter:
    def test_filter_optim_updates_parameters_with_meta_step(self):

        linear = nn.Linear(2, 2)
        before = get_model_params(linear)
        optim = ParamFilter(
            linear.parameters(), OptimFactory("SGD", lr=1e-2), OptimFactory("SGD", 1e-3)
        )
        optim.zero_grad()
        linear(torch.rand(3, 2)).sum().backward()
        optim.step()
        linear(torch.rand(3, 2)).sum().backward()
        optim.step()
        optim.adv()
        after = get_model_params(linear)
        assert (before != after).any()

    def test_transfer_copies_parameters_to_active(self):

        linear = nn.Linear(2, 2)
        before = get_model_params(linear)
        optim = ParamFilter(
            linear.parameters(), OptimFactory("SGD", lr=1e-2), OptimFactory("SGD", 1e-3)
        )
        optim.zero_grad()
        linear(torch.rand(3, 2)).sum().backward()
        optim.step()
        linear(torch.rand(3, 2)).sum().backward()
        optim.step()
        optim.transfer()
        after = get_model_params(linear)
        assert (before == after).all()

    def test_copy_meta_to_copies_to_new_module(self):

        linear = nn.Linear(2, 2)
        linear_test = nn.Linear(2, 2)
        before = get_model_params(linear_test)
        optim = ParamFilter(
            linear.parameters(), OptimFactory("SGD", lr=1e-2), OptimFactory("SGD", 1e-3)
        )
        optim.zero_grad()
        linear(torch.rand(3, 2)).sum().backward()
        optim.step()
        optim.step_filter()
        optim.copy_filter_optim_to(linear_test.parameters())
        after = get_model_params(linear_test)
        assert (before != after).any()

    def test_copy_meta_to_copies_to_new_tensor(self):

        x = torch.rand(2, 3, requires_grad=True)

        x_test = torch.rand(2, 3)
        before = torch.clone(x_test)
        optim = ParamFilter(
            [x], OptimFactory("SGD", lr=1e-2), OptimFactory("SGD", 1e-3)
        )
        optim.zero_grad()
        x.sum().backward()
        optim.step()
        optim.step_filter()
        optim.copy_filter_optim_to([x_test])
        assert (before != x_test).any()


class TestNullOptim:
    def test_null_optim_does_not_update_parameters(self):

        mod = nn.Linear(2, 2)
        before = get_model_params(mod)
        mod(torch.rand(4, 2)).mean().backward()
        optim = NullOptim(mod.parameters())
        optim.step()
        assert (before == get_model_params(mod)).all()

    def test_load_state_dict_works(self):

        mod = nn.Linear(2, 2)
        optim = NullOptim(mod.parameters())
        state_dict = optim.state_dict()
        optim.load_state_dict(state_dict)
        assert optim.state_dict() == state_dict


class TestOptimf:
    def test_null_optimf_creates_null_optim(self):

        mod = nn.Linear(2, 2)
        optimizer = optimf("NullOptim")(mod.parameters())
        assert isinstance(optimizer, NullOptim)

    def test_null_optimf_creates_sgd(self):

        mod = nn.Linear(2, 2)
        optimizer = optimf("SGD", lr=1e-3)(mod.parameters())
        assert isinstance(optimizer, optim.SGD)
