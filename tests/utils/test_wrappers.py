from zenkai.kaku import _wrappers
import torch
import torch.nn as nn


class TestGaussianGradHook:
    def test_gaussian_grad_hook_modifies_gradient(self):

        x = torch.randn(2, 4, requires_grad=True)
        y = torch.randn(4, 3)
        x_clone = x.clone()
        x_clone.requires_grad_()
        x_clone.retain_grad()
        hook = _wrappers.GaussianGradHook(1.0)
        x.register_hook(hook.grad_hook)
        z = x @ y
        z.register_hook(hook.grad_out_hook)
        z.mean().backward()

        (x_clone @ y).mean().backward()
        assert (x.grad != x_clone.grad).any()


class TestGradHookWrapper:
    def test_gaussian_grad_hook_modifies_gradient(self):

        x = torch.randn(2, 4, requires_grad=True)
        x_clone = x.clone().detach()
        x_clone.requires_grad_()
        x_clone.retain_grad()
        linear = nn.Linear(4, 3)

        wrapper = _wrappers.HookWrapper(linear, _wrappers.GaussianGradHook)
        y1 = wrapper(x)
        y2 = linear(x_clone)

        y1.mean().backward()
        y2.mean().backward()
        assert (x.grad != x_clone.grad).any()


class TestLambda:
    def test_lambda_outputs_same_result(self):

        torch.manual_seed(1)
        x = torch.rand(2, 2)
        lam = _wrappers.Lambda(lambda x: x + 1)
        assert (lam(x) == (x + 1)).all()

    def test_lambda_outputs_same_result_with_two_inputs(self):

        torch.manual_seed(1)
        x = torch.rand(2, 2)
        y = torch.rand(2, 2)
        lam = _wrappers.Lambda(lambda x, y: x + y)
        assert (lam(x, y) == (x + y)).all()

    def test_lambda_outputs_same_result_with_two_outputs(self):

        torch.manual_seed(1)
        x = torch.rand(2, 2)
        lam = _wrappers.Lambda(lambda x: (x + 1, x + 2))
        y1, y2 = lam(x)
        assert (y1 == (x + 1)).all()
        assert (y2 == (x + 2)).all()
