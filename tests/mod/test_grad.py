from zenkai.mod import _grad
import torch
import torch.nn as nn


class TestGaussianGradHook:

    def test_gaussian_grad_hook_modifies_gradient(self):

        x = torch.randn(2, 4, requires_grad=True)
        y = torch.randn(4, 3)
        x_clone = x.clone()
        x_clone.requires_grad_()
        x_clone.retain_grad()
        hook = _grad.GaussianGradHook(1.0)
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

        wrapper = _grad.HookWrapper(linear, _grad.GaussianGradHook)
        y1 = wrapper(x)
        y2 = linear(x_clone)

        y1.mean().backward()
        y2.mean().backward()
        assert (x.grad != x_clone.grad).any()
