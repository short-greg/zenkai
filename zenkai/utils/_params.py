
# 1st Party
import math
import typing

# 3rd party
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from itertools import chain
import pandas as pd
### These are the OLD functions


def set_model_grads(model: typing.Union[nn.Module, typing.Iterator[torch.nn.parameter.Parameter], torch.Tensor], theta_grad: typing.List[typing.Union[torch.Tensor, None]]):
    """Set the gradients of a module to the values specified by theta_grad

    Args:
        model (nn.Module): The module to update gradients for or a callable that returns parameters
        theta_grad (torch.Tensor): The gradient values to update with
    """
    model = get_p(model)
    for p, grad in zip(model, theta_grad):
        if grad is not None:
            grad = grad.detach()

        if p.grad is not None:
            with torch.no_grad():
                p.grad.copy_(grad)
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

    with torch.no_grad():

        for p, grad in zip(model, theta_grad):
            
            if grad is None and not to_add:
                p.grad = None
            
            if p.grad is None:
                if grad is not None:
                    p.grad = grad.clone()
            else:
                if grad is not None and to_add:
                    p.grad.add_(grad)
                elif grad is not None:
                    p.grad.copy_(grad)


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
                p.grad = grad.clone()
        else:
            if grad is not None and to_add:
                with torch.no_grad():
                    p.grad.add_(grad)
            elif grad is not None:
                with torch.no_grad():
                    p.grad.copy_(grad)


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
            with torch.no_grad():
                grad = p.grad.clone()
        else:
            grad = p.grad
        grads.append(grad)
    if len(grads) == 0:
        return None
    
    if flat_cat:
        return torch.concat([p.flatten() for p in grads])
    return grads


def model_params(models: typing.Iterable[nn.Module]) -> typing.Iterator:

    return chain(model.parameters() for model in models)


### These are the NEW functions

PObj = typing.Union[nn.Module, typing.Iterator[torch.nn.parameter.Parameter], torch.Tensor, typing.Callable[[],typing.Iterator[torch.nn.parameter.Parameter]]]


def get_p(obj: PObj) -> typing.Iterable[torch.nn.parameter.Parameter]:
    
    if isinstance(obj, nn.Module):
        return obj.parameters()
    elif isinstance(obj, torch.Tensor):
        return [obj]
    elif isinstance(obj, typing.Callable):
        return obj()
    # assume it is an iterable
    elif isinstance(obj, typing.Iterator):
        return obj
    else:
        result = []
        for p in obj:

            if isinstance(p, typing.Iterator):
                result.append(p) 
            elif isinstance(p, nn.Module):
                result.append(p.parameters())
            elif isinstance(p, typing.Callable):
                result.append(p())
            else:
                result.append([p])
    return chain(*result)


def get_grad(obj: PObj) -> typing.Iterator[torch.nn.parameter.Parameter]:
    
    for p in get_p(obj):
        return p.grad


def to_pvec(obj: PObj) -> torch.Tensor:

    return torch.cat([p_i.flatten() for p_i in get_p(obj)], dim=0)


def to_gradvec(obj: PObj) -> torch.Tensor:

    result = []
    for p_i in get_p(obj):
        if p_i.grad is None:
            result.append(torch.zeros_like(p_i).flatten())
        else:
            result.append(p_i.grad.flatten())
    return torch.cat(result, dim=0)


def align_vec(obj: PObj, vec: torch.Tensor) -> typing.Iterator[typing.Tuple[torch.Tensor, torch.Tensor]]:
    start = 0
    for p in get_p(obj):

        end = start + p.numel()
        cur_vec = vec[start:end]
        cur_vec = cur_vec.reshape(p.shape)
        start = end
        yield p, cur_vec


def set_pvec(obj: PObj, vec: torch.Tensor):

    for p, cur_vec in align_vec(obj, vec):
        set_params(p, cur_vec)


def acc_pvec(obj: PObj, vec: torch.Tensor):

    for p, cur_vec in align_vec(obj, vec):
        acc_params(p, cur_vec)


def set_gradvec(obj: PObj, vec: torch.Tensor):
    
    for p, cur_vec in align_vec(obj, vec):
        set_grad(p, cur_vec)


def acc_gradvec(obj: PObj, vec: torch.Tensor):

    for p, cur_vec in align_vec(obj, vec):
        acc_grad(p, cur_vec)


def set_gradtvec(obj: PObj, vec: torch.Tensor):
    
    for p, cur_vec in align_vec(obj, vec):
        set_gradt(p, cur_vec)


def acc_gradtvec(obj: PObj, vec: torch.Tensor):

    for p, cur_vec in align_vec(obj, vec):
        acc_gradt(p, cur_vec)

    # return torch.cat([p_i.flatten() for p_i in get_p(obj)], dim=0)


def get_params(model: nn.Module) -> torch.Tensor:
    """Convenience function to retrieve the parameters of a model

    Args:
        model (nn.Module): 

    Returns:
        torch.Tensor: 
    """
    
    try:
        p = get_p(model)
        return parameters_to_vector(p)
    except NotImplementedError:
        return None


def to_df(name: str, obj: PObj) -> pd.DataFrame:

    return pd.DataFrame(
        {name: [p for p in get_p(obj)]}
    )


def to_series(obj: PObj) -> pd.DataFrame:

    return pd.Series(
        [p for p in get_p(obj)]
    )


def get_multp(objs: typing.Iterable[PObj]) -> typing.Tuple[torch.nn.parameter.Parameter]:
    """Get params in a tuple. Primarily to 
    make it easier to zip multiple modules

    Args:
        objs (typing.Iterable[PObj]): The parameter objects to get

    Returns:
        typing.Iterable[typing.Tuple[torch.nn.parameter.Parameter]]: The tuple of parameters
    """

    return tuple(
        get_p(obj) for obj in objs
    )


def loop_p(obj: PObj) -> typing.Iterator[torch.nn.parameter.Parameter]:
    """

    Args:
        obj (PObj): The parameter object to loop over

    Returns:
        typing.Iterator[torch.nn.parameter.Parameter]: 

    Yields:
        Iterator[typing.Iterator[torch.nn.parameter.Parameter]]: The parameters
    """

    for p in get_p(obj):
        yield p


def apply_p(
    obj: PObj, f
):
    """Apply a function to the parameters

    Args:
        parameters (typing.Iterator[torch.nn.parameter.Parameter]): Parameters to apply a function to
        f : The function to apply
    """
    with torch.no_grad():
        for p in get_p(obj):
            
            p.copy_(f(p))


def apply_grad(
    obj: PObj, f: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], skip_none: bool=True
):
    """Apply a function to the parameters

    Args:
        parameters (typing.Iterator[torch.nn.parameter.Parameter]): Parameters to apply a function to
        f : The function to apply
    """
    with torch.no_grad():
        for p in get_p(obj):
            if p.grad is None and skip_none:
                continue
            elif p.grad is None:
                p.grad = f(p, p.grad).clone()
            else:
                p.grad.copy_(f(p, p.grad))


def set_params(
    cur: torch.Tensor, new_: torch.Tensor
):
    with torch.no_grad():
        cur.copy_(new_.detach())


def acc_params(
    cur: torch.Tensor, new_: torch.Tensor
):
    with torch.no_grad():
        cur.copy_(cur + new_)


def set_grad(
    cur: torch.Tensor, grad: torch.Tensor
):
    with torch.no_grad():
        if cur.grad is None:
            cur.grad = grad.clone()
        else:
            with torch.no_grad():
                cur.grad.copy_(grad.detach())
    

def set_gradt(
    cur: torch.Tensor, t: torch.Tensor
):
    grad = cur - t
    with torch.no_grad():
        cur.grad.copy_(grad.detach())


def acc_grad(
    cur: torch.Tensor, grad: torch.Tensor
):
    with torch.no_grad():
        if cur.grad is None:
            cur.grad = grad.clone()
        else:
            with torch.no_grad():
                cur.grad.copy_(cur.grad + grad)


def acc_gradt(
    cur: torch.Tensor, t: torch.Tensor
):
    grad = cur - t
    with torch.no_grad():
        if cur.grad is None:
            cur.grad = grad.clone()
        else:
            with torch.no_grad():
                cur.grad.add_(grad)


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


def reg_p(obj: PObj, f) -> torch.Tensor:
    """Convenience function to regularize parameters

    Args:
        obj (PObj): The parameter object ot regularize

    Returns:
        torch.Tensor: the regularization value
    """
    regularization = None
    for p in get_p(obj):
        
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
                    with torch.no_grad():
                        value.grad.copy_(stored)
                elif value.grad is not None:
                    value.grad = None
                else:
                    value.grad = stored
            else:
                update_model_grads(
                    value, stored, False
                )
