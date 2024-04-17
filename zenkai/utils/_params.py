
# 1st Party
import math
import typing

# 3rd party
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from itertools import chain


def get_model_params(model: nn.Module) -> torch.Tensor:
    """Convenience function to retrieve the parameters of a model

    Args:
        model (nn.Module): 

    Returns:
        torch.Tensor: 
    """
    try:
        return parameters_to_vector(model.parameters())
    except NotImplementedError:
        return None


def model_params(models: typing.Iterable[nn.Module]) -> typing.Iterator:

    return chain(model.parameters() for model in models)


PObj = typing.Union[nn.Module, typing.Iterator[torch.nn.parameter.Parameter], torch.Tensor, typing.Callable[[],typing.Iterator[torch.nn.parameter.Parameter]]]


def get_params(mod: PObj) -> typing.Iterable[torch.nn.parameter.Parameter]:
    
    if isinstance(mod, nn.Module):
        return mod.parameters()
    if isinstance(mod, torch.Tensor):
        return [mod]
    if isinstance(mod, typing.Callable):
        return mod()
    # assume it is an iterable
    if isinstance(mod, typing.Iterator):
        return mod
    result = []
    for p in mod:

        if isinstance(p, typing.Iterator):
            result.append(p) 
        elif isinstance(p, nn.Module):
            result.append(p.parameters())
        else:
            result.append([p])
    return chain(*result)
    

def apply_to_params(
    parameters: typing.Iterator[torch.nn.parameter.Parameter], f
):
    """Apply a function to the parameters

    Args:
        parameters (typing.Iterator[torch.nn.parameter.Parameter]): Parameters to apply a function to
        f : The function to apply
    """
    if isinstance(parameters, nn.Module):
        parameters = parameters.parameters()
    
    elif isinstance(parameters, typing.List):
        parameters = chain(
            *(parameter.parameters() if isinstance(parameter, nn.Module) else parameter for parameter in parameters)
        )

    for p in parameters:
        p.data = f(p.data)


def update_model_params(
        model: typing.Union[nn.Module, typing.Iterator[torch.nn.parameter.Parameter]], theta: torch.Tensor):
    """Convenience function to update the parameters of a model

    Args:
        model (nn.Module): Model to update parameters for
        theta (torch.Tensor): The new parameters for the model
    """
    if isinstance(model, torch.nn.Module):
        model = model.parameters()
    vector_to_parameters(theta, model)


def set_model_grads(model: typing.Union[nn.Module, typing.Iterator[torch.nn.parameter.Parameter], torch.Tensor], theta_grad: typing.List[typing.Union[torch.Tensor, None]]):
    """Set the gradients of a module to the values specified by theta_grad

    Args:
        model (nn.Module): The module to update gradients for or a callable that returns parameters
        theta_grad (torch.Tensor): The gradient values to update with
    """
    model = get_params(model)
    for p, grad in zip(model, theta_grad):
        if grad is not None:
            grad = grad.detach()

        if p.grad is not None:
            p.grad.data = grad
        else:
            p.grad = grad


def update_model_grads(model: typing.Union[nn.Module, typing.Iterator[torch.nn.parameter.Parameter], torch.Tensor], theta_grad: typing.List[typing.Union[torch.Tensor, None]], to_add: bool = True):
    """Update the gradients of a module

    Args:
        model (nn.Module): The module to update gradients for
        theta_grad (torch.Tensor): The gradient values to update with
        to_add (bool): Whether to add the new gradients to the current ones or to replace the gradients
    """
    # start = 0
    if isinstance(model, nn.Module):
        model = model.parameters()
    elif isinstance(model, torch.Tensor):
        model = [model]

    for p, grad in zip(model, theta_grad):
        
        if grad is None and not to_add:
            p.grad = None
        
        if p.grad is None:
            if grad is not None:
                p.grad = grad.detach()
        else:
            if grad is not None and to_add:
                p.grad.data = p.grad.data + grad.detach()
            elif grad is not None:
                p.grad.data = grad.detach() 


def update_model_grads_with_t(model: typing.Union[nn.Module, typing.Iterator[torch.nn.parameter.Parameter], torch.Tensor], t: typing.List[typing.Union[torch.Tensor, None]], to_add: bool = True, lr: float=1.0):
    """Update the gradients of a module

    Args:
        model (nn.Module): The module to update gradients for
        theta_grad (torch.Tensor): The gradient values to update with
        to_add (bool): Whether to add the new gradients to the current ones or to replace the gradients
    """
    # start = 0
    if isinstance(model, nn.Module):
        model = model.parameters()
    elif isinstance(model, torch.Tensor):
        model = [model]

    for p, t_i in zip(model, t):
        
        grad = ((p - t_i) * lr).detach()
        if grad is None and not to_add:
            p.grad = None
        
        if p.grad is None:
            if grad is not None:
                p.grad = grad.detach()
        else:
            if grad is not None and to_add:
                p.grad.data = p.grad.data + grad.detach()
            elif grad is not None:
                p.grad.data = grad.detach() 


MODEL_P = typing.Union[nn.Module, typing.Iterator[torch.nn.parameter.Parameter], torch.Tensor]


def get_model_grads(
    model: MODEL_P, clone: bool=True, flat_cat: bool=False
) -> typing.Union[typing.List[torch.Tensor], torch.Tensor, None]:
    """Get all of the gradients in a module

    Args:
        model (nn.Module): the module to get grads for
        clone: Whether to clone the gradient
        flat_cat: Whether to flatten and concatenate the output

    Returns:
        torch.Tensor or None: the grads flattened. Returns None if any of the grads have not been set
    """

    grads = []
    if isinstance(model, nn.Module):
        model = model.parameters()
    elif isinstance(model, torch.Tensor):
        model = [model]

    for p in model:
        if p.grad is not None and clone:
            grad = p.grad.data.clone()
        else:
            grad = p.grad
        grads.append(grad)
    if len(grads) == 0:
        return None
    
    if flat_cat:
        return torch.concat([p.flatten() for p in grads])
    return grads


def regularize_params(model: MODEL_P, f) -> torch.Tensor:
    """Convenience function to regularize parameters

    Args:
        model (MODEL_P): The model or parameters to regularize
        f: regularization function

    Returns:
        torch.Tensor: the regularization value
    """

    regularization = None
    if isinstance(model, nn.Module):
        model = model.parameters()
    elif isinstance(model, torch.Tensor):
        model = [model]

    for p in model:
        
        cur = f(p)
        if cur.dim() != 0:
            raise RuntimeError('The regularization function did not output a reduced value of dim 0')
        if regularization is None:
            regularization = cur
        else:
            regularization = regularization + cur
    return p


class undo_grad(object):

    def __init__(
        self, values: typing.Iterable[typing.Union[typing.Callable[[], typing.Iterator], nn.Module, torch.Tensor, nn.parameter.Parameter]]
    ):
        self._values = values
        self._stored = []

    def __enter__(self):

        for value in self._values:
            if (
                isinstance(value, torch.Tensor) or isinstance(value, torch.nn.parameter.Parameter)
            ):
                self._stored.append(
                    value.grad.clone() if value.grad is not None else None
                )
            else:
                self._stored.append(
                    get_model_grads(value)
                )
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        
        if exc_type is not None:
            exc_val
        for stored, value in zip(self._stored, self._values):
            
            if (
                isinstance(value, torch.Tensor) or isinstance(value, torch.nn.parameter.Parameter)
            ):
                if value.grad is not None and stored is not None:
                    value.grad.data = stored
                elif value.grad is not None:
                    value.grad = None
                else:
                    value.grad = stored
            else:
                update_model_grads(
                    value, stored, False
                )
        