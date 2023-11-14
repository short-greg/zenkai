# 1st party
from abc import abstractmethod
import typing
from functools import partial

# 3rd party
import torch
import torch.nn as nn


class GradHook(object):
    """Define a wrapper that modifies the gradeint of a module"""

    def __init__(self):
        """Create a grad hook wrapper"""
        self.grad_out = None

    def grad_out_hook(self, grad: torch.Tensor):
        """

        Args:
            grad (torch.Tensor): _description_
        """
        self.grad_out = grad

    @abstractmethod
    def grad_hook(self, grad: torch.Tensor):
        pass


class GaussianGradHook(GradHook):
    """ """

    def __init__(self, weight: float = 1.0):
        """Create

        Args:
            weight (float, optional): . Defaults to 1.0.
        """
        super().__init__()
        self.weight = weight

    def grad_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """Method that decorates the gradient

        Args:
            grad (torch.Tensor):

        Returns:
            torch.Tensor: the gradient with noise added that is a function of the output
        """
        grad_out = (
            self.grad_out * torch.randn_like(self.grad_out) * self.weight
        ).unsqueeze(-2)

        grad = (grad.unsqueeze(-1) + grad_out).mean(dim=-1)
        return grad

    @classmethod
    def factory(self, weight: float = 1.0) -> typing.Callable[[], "GaussianGradHook"]:

        return partial(GaussianGradHook, weight=weight)


class HookWrapper(nn.Module):
    def __init__(self, wrapped: nn.Module, grad_hook_factory: typing.Type[GradHook]):
        """

        Args:
            wrapped (nn.Module):
            grad_hook_factory (typing.Type[GradHook]):
        """
        super().__init__()
        self.grad_hook_factory = grad_hook_factory
        self.wrapped = wrapped

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        grad_hook = self.grad_hook_factory()
        if self.training:
            x.register_hook(grad_hook.grad_hook)
            x = self.wrapped(x)
            x.register_hook(grad_hook.grad_out_hook)
        else:
            x = self.wrapped(x)
        return x


class Lambda(nn.Module):
    """
    A generic function
    """

    def __init__(self, f, *args, **kwargs):
        super().__init__()
        self._f = f
        self._args = args
        self._kwargs = kwargs

    def forward(self, *x: torch.Tensor):
        return self._f(*x, *self._args, **self._kwargs)
