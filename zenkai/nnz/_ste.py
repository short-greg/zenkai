# 3rd party
import torch


class SignSTE(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x):
        """Forward pass of the Binary Step function."""
        ctx.save_for_backward(x)
        return torch.sign(x)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of the Binary Step function using the Straight-Through Estimator."""
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
        """Forward pass of the Binary Step function."""
        ctx.save_for_backward(x)
        return torch.clamp(x, 0, 1).round()

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass of the Binary Step function using the Straight-Through Estimator."""
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x < 0) | (x > 1)] = 0
        return grad_input
