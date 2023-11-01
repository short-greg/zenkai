# 3rd party
import torch
import torch.nn as nn
import torch.nn.functional


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



class FreezeDropout(nn.Module):
    """Freeze the dropout
    """

    def __init__(self, p: float, freeze: bool=False):
        """Create a FreezeDropout

        Args:
            p (float): The dropout rate
            freeze (bool, optional): Whether to freeze the dropout. Defaults to False.

        Raises:
            ValueError: If p is greater or equal to one or less than zero
        """
        super().__init__()
        if p >= 1.0 or p < 0.0:
            raise ValueError(f'P must be in range [0.0, 1.0) not {p}')
        self.p = p
        self.freeze = freeze
        self._cur = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.p == 0.0:
            return x

        if not self.training:
            return x * (1 / 1 - self.p)

        if self.freeze and self._cur is not None:
            f = self._cur
        else:
            f = (torch.rand_like(x) > self.p).type_as(x)
        
        self._cur = f
        return f * x


# TODO: Remove

class Argmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self._dim = dim

    def forward(self, x: torch.Tensor) -> torch.LongTensor:
        return torch.argmax(x, dim=-1)


class Sign(nn.Module):

    def __init__(self, grad: bool = True):
        super().__init__()
        self._grad = grad

    def forward(self, x: torch.Tensor):
        if self._grad:
            return SignSTE.apply(x)
        return torch.sign(x)


class Binary(nn.Module):

    def __init__(self, grad: bool = True):
        super().__init__()
        self._grad = grad

    def forward(self, x: torch.Tensor):
        if self._grad:
            return BinarySTE.apply(x)
        return torch.clamp(x, 0, 1).round()


class SignSTE(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x):
        """
        Forward pass of the Binary Step function.
        """
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # return grad_input.clamp(-1, 1)
        grad_input[(x < -1) | (x > 1)] = 0
        return grad_input


class BinarySTE(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x):
        """
        Forward pass of the Binary Step function.
        """
        ctx.save_for_backward(x)
        return torch.clamp(x, 0, 1).round()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # return grad_input.clamp(-1, 1)
        grad_input[(x < -1) | (x > 1)] = 0
        return grad_input


class Clamp(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x, lower: float=0, upper: float=1, backward_lower: float=-1, backward_upper: float=1):
        """
        Forward pass of the Binary Step function.
        """
        ctx.save_for_backward(x)
        ctx.lower = lower
        ctx.upper = upper
        ctx.backward_lower = backward_lower
        ctx.backward_upper = backward_upper
        return torch.clamp(x, lower, upper)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the Binary Step function using the Straight-Through Estimator.
        """
        grad_input = grad_output.clone()
        return grad_input.clamp(ctx.backward_lower, ctx.backward_upper), None, None, None, None



def binary_ste(x: torch.Tensor) -> torch.Tensor:
    return BinarySTE.apply(x)


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    return SignSTE.apply(x)
