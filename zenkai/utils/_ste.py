# 3rd party
import torch


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
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x < -1) | (x > 1)] = 0
        return grad_input


class StepSTE(torch.autograd.Function):
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
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x < 0) | (x > 1)] = 0
        return grad_input


def step_ste(x: torch.Tensor) -> torch.Tensor:
    """

    Args:
        x (torch.Tensor): The input

    Returns:
        torch.Tensor: 0 for values less or equal to 0 otherwise 1
    """
    return StepSTE.apply(x)


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    """Execute the sign function

    Args:
        x (torch.Tensor): the input

    Returns:
        torch.Tensor: -1 for values less than 0 otherwise 1
    """
    return SignSTE.apply(x)
