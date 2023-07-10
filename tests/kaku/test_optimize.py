# 1st Party
from functools import partial

# 3rd Party
import torch
from torch import nn
from torch import optim as torch_optim
from torch.nn import utils

# Local
from zenkai.kaku.optimize import MetaOptim, OptimFactory
from zenkai.utils import get_model_parameters


class TestMetaOptim:

    def test_meta_optim_updates_parameters_with_meta_step(self):

        linear = nn.Linear(2, 2)
        before = get_model_parameters(linear)
        optim = MetaOptim(
            linear.parameters(), OptimFactory("sgd", lr=1e-2),
            OptimFactory("sgd", 1e-3)
        )
        optim.zero_grad()
        linear(torch.rand(3, 2)).sum().backward()
        optim.step()
        linear(torch.rand(3, 2)).sum().backward()
        optim.step()
        optim.adv()
        after = get_model_parameters(linear)
        assert (before != after).any()

    def test_transfer_copies_parameters_to_active(self):

        linear = nn.Linear(2, 2)
        before = get_model_parameters(linear)
        optim = MetaOptim(
            linear.parameters(), OptimFactory("sgd", lr=1e-2),
            OptimFactory("sgd", 1e-3)
        )
        optim.zero_grad()
        linear(torch.rand(3, 2)).sum().backward()
        optim.step()
        linear(torch.rand(3, 2)).sum().backward()
        optim.step()
        optim.transfer()
        after = get_model_parameters(linear)
        assert (before == after).all()

    def test_copy_meta_to_copies_to_new_module(self):

        linear = nn.Linear(2, 2)
        linear_test = nn.Linear(2, 2)
        before = get_model_parameters(linear_test)
        optim = MetaOptim(
            linear.parameters(), OptimFactory("sgd", lr=1e-2),
            OptimFactory("sgd", 1e-3)
        )
        optim.zero_grad()
        linear(torch.rand(3, 2)).sum().backward()
        optim.step()
        optim.step_meta()
        optim.copy_meta_to(linear_test.parameters())
        after = get_model_parameters(linear_test)
        assert (before != after).any()

    def test_copy_meta_to_copies_to_new_tensor(self):

        x = torch.rand(2, 3, requires_grad=True)
        
        x_test = torch.rand(2, 3)
        before = torch.clone(x_test)
        optim = MetaOptim(
            [x], OptimFactory("sgd", lr=1e-2),
            OptimFactory("sgd", 1e-3)
        )
        optim.zero_grad()
        x.sum().backward()
        optim.step()
        optim.step_meta()
        optim.copy_meta_to([x_test])
        assert (before != x_test).any()
