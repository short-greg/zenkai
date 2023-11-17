# 1st Party
import math
import typing

# 3rd party
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from itertools import chain

# TODO: Organize better


def to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def to_th(
    x: np.ndarray,
    dtype: torch.dtype,
    device: torch.device = None,
    requires_grad: bool = False,
    retains_grad: bool = False,
) -> torch.Tensor:
    """

    Args:
        x (np.ndarray): Array to convert
        dtype (torch.dtype): type to convert to
        device (torch.device): device to convert to
        requires_grad (bool, optional): Whether the tensor requires grad. Defaults to False.
        retains_grad (bool, optional): Whether the tensor retains a grad. Defaults to False.

    Returns:
        torch.Tensor: result
    """
    x: torch.Tensor = torch.tensor(
        x, dtype=dtype, requires_grad=requires_grad, device=device
    )
    if retains_grad:
        x.retain_grad()
    return x


def to_th_as(
    x: np.ndarray,
    as_: torch.Tensor,
    requires_grad: bool = False,
    retains_grad: bool = False,
) -> torch.Tensor:
    """

    Args:
        x (np.ndarray): Array to convert
        as_ (torch.Tensor): The array to base conversion off of
        requires_grad (bool, optional): Whether the tensor requires grad. Defaults to False.
        retains_grad (bool, optional): Whether the tensor retains a grad. Defaults to False.

    Returns:
        torch.Tensor: result
    """

    x: torch.Tensor = torch.tensor(
        x, dtype=as_.dtype, requires_grad=requires_grad, device=as_.device
    )
    if retains_grad:
        x.retain_grad()
    return x


def expand_k(x: torch.Tensor, k: int, reshape: bool = True) -> torch.Tensor:
    """expand the trial dimension in the tensor (separates the trial dimension from the sample dimension)

    Args:
        x (torch.Tensor): The tensor to update
        k (int): The number of trials
        reshape (bool, optional): Whether to use reshape (True) or view (False). Defaults to True.

    Returns:
        torch.Tensor: The expanded tensor
    """
    shape = torch.Size([k, -1, *x.shape[1:]])
    if reshape:
        return x.reshape(shape)
    return x.view(shape)


def collapse_k(x: torch.Tensor, reshape: bool = True) -> torch.Tensor:
    """collapse the trial dimension in the tensor (merges the trial dimension with the sample dimension)

    Args:
        x (torch.Tensor): The tensor to update
        reshape (bool, optional): Whether to use reshape (True) or view (False). Defaults to True.

    Returns:
        torch.Tensor: The collapsed tensor
    """
    if reshape:
        return x.reshape(-1, *x.shape[2:])
    return x.view(-1, *x.shape[2:])


def expand_dim0(x: torch.Tensor, k: int, reshape: bool = False) -> torch.Tensor:
    """Expand an input to repeat k times

    Args:
        x (torch.Tensor): input tensor
        k (int): Number of times to repeat. Must be greater than 0
        reshape (bool, optional): Whether to reshape the output so the first 
            and second dimensions are combined. Defaults to False.

    Raises:
        ValueError: If k is less than or equal to 0

    Returns:
        torch.Tensor: the expanded tensor
    """
    if k <= 0:
        raise ValueError(f"Argument k must be greater than 0 not {k}")

    y = x[None]

    y = y.repeat(k, *([1] * len(y.shape[1:])))  # .transpose(0, 1)
    if reshape:
        return y.view(y.shape[0] * y.shape[1], *y.shape[2:])
    return y


def flatten_dim0(x: torch.Tensor):
    """Flatten the population and batch dimensions of a population"""
    if x.dim() < 2:
        return x
    return x.view(x.shape[0] * x.shape[1], *x.shape[2:])


def deflatten_dim0(x: torch.Tensor, k: int) -> torch.Tensor:
    """Deflatten the population and batch dimensions of a population"""
    if x.dim() == 0:
        raise ValueError("Input dimension == 0")

    return x.view(k, -1, *x.shape[1:])


def freshen(x: torch.Tensor, requires_grad: bool = True, inplace: bool = False):
    if not isinstance(x, torch.Tensor):
        return x
    if inplace:
        x.detach_()
    else:
        x = x.detach()
    if requires_grad:
        x = x.requires_grad_(requires_grad)
        x.retain_grad()
    return x


def get_model_parameters(model: nn.Module) -> torch.Tensor:
    """Convenience function to retrieve the parameters of a model

    Args:
        model (nn.Module): _description_

    Returns:
        torch.Tensor: _description_
    """
    try:
        return parameters_to_vector(model.parameters())
    except NotImplementedError:
        return None


def model_parameters(models: typing.Iterable[nn.Module]) -> typing.Iterator:

    return chain(model.parameters() for model in models)


def apply_to_parameters(parameters: typing.Iterator[torch.nn.parameter.Parameter], f):
    """Apply a function to the parameters

    Args:
        parameters (typing.Iterator[torch.nn.parameter.Parameter]): Parameters to apply a function to
        f : The function to apply
    """
    if isinstance(parameters, typing.List):
        parameters = chain(*parameters)
    for p in parameters:
        p.data = f(p.data)


def update_model_parameters(model: nn.Module, theta: torch.Tensor):
    """Convenience function to update the parameters of a model

    Args:
        model (nn.Module): Model to update parameters for
        theta (torch.Tensor): The new parameters for the model
    """
    vector_to_parameters(theta, model.parameters())


def set_model_grads(model: nn.Module, theta_grad: torch.Tensor):
    """Set the gradients of a module to the values specified by theta_grad

    Args:
        model (nn.Module): The module to update gradients for
        theta_grad (torch.Tensor): The gradient values to update with
    """
    start = 0
    for p in model.parameters():
        finish = start + p.numel()
        cur = theta_grad[start:finish].reshape(p.shape)
        p.grad = cur.detach()
        start = finish


def update_model_grads(model: nn.Module, theta_grad: torch.Tensor, to_add: bool = True):
    """Update the gradients of a module

    Args:
        model (nn.Module): The module to update gradients for
        theta_grad (torch.Tensor): The gradient values to update with
        to_add (bool): Whether to add the new gradients to the current ones or to replace the gradients
    """
    start = 0
    for p in model.parameters():
        finish = start + p.numel()
        cur = theta_grad[start:finish].reshape(p.shape)
        if p.grad is None or not to_add:
            p.grad = cur.detach()
        elif to_add:
            p.grad.data = p.grad.data + cur.detach()
        start = finish


def get_model_grads(model: nn.Module) -> typing.Union[torch.Tensor, None]:
    """Get all of the gradients in a module

    Args:
        model (nn.Module): the module to get grads for

    Returns:
        torch.Tensor or None: the grads flattened. Returns None if any of the grads have not been set
    """

    grads = []
    for p in model.parameters():
        if p.grad is None:
            return None
        grads.append(p.grad.flatten())
    if len(grads) == 0:
        return None
    return torch.cat(grads)


def lr_update(
    current: torch.Tensor, new_: torch.Tensor, lr: typing.Optional[float] = None
) -> torch.Tensor:
    """update tensor with learning rate

    Args:
        current (torch.Tensor): current tensor
        new_ (torch.Tensor): the new tensor
        lr (typing.Optional[float], optional): the learning rate. Defaults to None.

    Returns:
        torch.Tensor: the updated tensor
    """
    assert lr is None or (0.0 <= lr <= 1.0)
    if lr is not None:
        new_ = (lr * new_) + (1 - lr) * (current)
    return new_


def lr_update_param(
    current: torch.Tensor, new_: torch.Tensor, lr: typing.Optional[float] = None
) -> nn.parameter.Parameter:
    """update tensor with learning rate

    Args:
        current (torch.Tensor): current tensor
        new_ (torch.Tensor): the new tensor
        lr (typing.Optional[float], optional): the learning rate. Defaults to None.

    Returns:
        nn.parameter.Parameter: the updated tensor as a parameter
    """
    p = nn.parameter.Parameter(lr_update(current, new_, lr).detach())
    return p


def to_zero_neg(x: torch.Tensor) -> torch.Tensor:
    """convert a 'signed' binary tensor to have zeros for negatives

    Args:
        x (torch.Tensor): Signed binary tensor. Tensor must be all -1s or 1s to get expected result

    Returns:
        torch.Tensor: The binary tensor with negatives as zero
    """

    return (x + 1) / 2


def to_signed_neg(x: torch.Tensor) -> torch.Tensor:
    """convert a 'zero' binary tensor to have negative ones for negatives

    Args:
        x (torch.Tensor): Binary tensor with zeros for negatives. 
            Tensor must be all zeros and ones to get expected result

    Returns:
        torch.Tensor: The signed binary tensor
    """
    return (x * 2) - 1


def binary_encoding(
    x: torch.LongTensor, n_size: int, bit_size: bool = False
) -> torch.Tensor:
    """Convert an integer tensor to a binary encoding

    Args:
        x (torch.LongTensor): The integer tensor
        n_size (int): The size of the encoding (e.g. number of bits if using bits or the max number)
        bit_size (bool, optional): Whether the size is described in terms of number of bits . Defaults to False.

    Returns:
        torch.Tensor: The binary encoding
    """

    if not bit_size:
        n_size = int(math.ceil(math.log2(n_size)))
    results = []
    for _ in range(n_size):
        results.append(x)
        x = x >> 1
    results = torch.stack(tuple(reversed(results))) & 1
    shape = list(range(results.dim()))
    shape = shape[1:] + shape[0:1]
    return results.permute(*shape)


def module_factory(module: typing.Union[str, nn.Module], *args, **kwargs) -> nn.Module:

    if isinstance(module, nn.Module):
        if len(args) != 0:
            raise ValueError("Cannot set args if module is already defined")
        if len(kwargs) != 0:
            raise ValueError("Cannot set kwargs if module is already defined")

        return module

    return getattr(nn, module)(*args, **kwargs)


def unsqueeze_to(source: torch.Tensor, align_to: torch.Tensor) -> torch.Tensor:
    """Unsqueeze a tensor to align with another tensor that has more dimensions
    Will only work if source has fewer dimensions than align to and all of those dimensions
    are already aligned

    Args:
        source (torch.Tensor): the tensor to unsqueeze
        align_to (torch.Tensor): the tensor to align to

    Returns:
        torch.Tensor: the aligned tensor
    """

    for i in range(source.dim(), align_to.dim()):
        source = source.unsqueeze(i)
    return source


def align_to(source: torch.Tensor, align_to: torch.Tensor) -> torch.Tensor:
    """Unsqueeze a tensor to align with another tensor that has more dimensions
    Will only work if source has fewer dimensions than align to and all of those dimensions
    are already aligned

    Args:
        source (torch.Tensor): the tensor to unsqueeze
        align_to (torch.Tensor): the tensor to align to

    Returns:
        torch.Tensor: the aligned tensor
    """
    shape = [1] * source.dim()
    for i in range(source.dim(), align_to.dim()):
        source = source.unsqueeze(i)
        shape.append(align_to.shape[i])
    source = source.repeat(*shape)
    return source


def decay(
    new_v: torch.Tensor,
    cur_v: typing.Union[torch.Tensor, float, None] = None,
    decay: float = 0.1,
) -> torch.Tensor:
    """Decay the current

    Args:
        new_v (torch.Tensor): The new value
        cur_v (typing.Union[torch.Tensor, float, None], optional): The current value. Defaults to None.
        decay (float, optional): The amount to reduce the current . Defaults to 0.1.

    Returns:
        torch.Tensor: The updated tensor
    """
    if cur_v is None or decay == 0.0:
        return new_v
    return decay * cur_v + (1 - decay) * new_v


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
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        # return grad_input.clamp(-1, 1)
        grad_input[(x < -1) | (x > 1)] = 0
        return grad_input


def binary_ste(x: torch.Tensor) -> torch.Tensor:
    return BinarySTE.apply(x)


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    return SignSTE.apply(x)


# def calc_correlation_mae(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
#     """Calculate the mean absolute error in correlation

#     Args:
#         x1 (torch.Tensor)
#         x2 (torch.Tensor)

#     Returns:
#         torch.Tensor: The correlation MAE
#     """

#     corr1 = torch.corrcoef(torch.flatten(x1, 1))
#     corr2 = torch.corrcoef(torch.flatten(x2, 1))
#     return torch.abs(corr1 - corr2).mean()
