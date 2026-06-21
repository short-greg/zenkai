# 1st Party
import typing
from itertools import chain

# 3rd Party
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

PObj = typing.Union[
    nn.Module,
    typing.Iterator[torch.nn.parameter.Parameter],
    torch.Tensor,
    typing.Callable[[], typing.Iterator[torch.nn.parameter.Parameter]],
]


def p_get(obj: PObj) -> typing.Iterable[torch.nn.parameter.Parameter]:
    """Get all of the parameters for a "PObj"

    Args:
        obj (PObj): The parameter object to get the parameters for

    Returns:
        typing.Iterable[torch.nn.parameter.Parameter]: An iterable object to loop over the parameters
    """

    if isinstance(obj, nn.Module):
        return obj.parameters()
    elif isinstance(obj, torch.Tensor):
        return [obj]
    elif isinstance(obj, typing.Callable):
        return obj()
    # assume it is an iterable
    elif isinstance(obj, typing.Iterator):
        return obj
    # assume it is a list
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


def grad_get(obj: PObj) -> typing.Iterator[torch.nn.parameter.Parameter]:
    """Get the gradients for a PObj

    Args:
        obj (PObj): The PObj to get parameters for

    yields:
        torch.nn.parameter.Parameter: All the gradients for tetheh PObj
    """

    for p in p_get(obj):
        yield p.grad


def to_pvec(obj: PObj) -> torch.Tensor:
    """Convert a PObj to a flattened vector

    Args:
        obj (PObj): The object to convert

    Returns:
        torch.Tensor: The vector
    """

    return torch.cat([p_i.flatten() for p_i in p_get(obj)], dim=0)


def to_gradvec(obj: PObj) -> torch.Tensor:
    """Retrieve a vector of gradients from the grad object

    Args:
        obj (PObj): The object to convert

    Returns:
        torch.Tensor: The Gradient
    """

    result = []
    for p_i in p_get(obj):
        if p_i.grad is None:
            result.append(torch.zeros_like(p_i).flatten())
        else:
            result.append(p_i.grad.flatten())
    return torch.cat(result, dim=0)


def pvec_align(obj: PObj, vec: torch.Tensor) -> typing.Iterator[typing.Tuple[torch.Tensor, torch.Tensor]]:
    """Align a vector to parmaters for a PObject

    Args:
        obj (PObj): The PObject to align to
        vec (torch.Tensor): The vector to align


    Yields:
        Iterator[typing.Iterator[typing.Tuple[torch.Tensor, torch.Tensor]]]: The aligned vectors
    """
    start = 0
    for p in p_get(obj):

        end = start + p.numel()
        cur_vec = vec[start:end]
        cur_vec = cur_vec.reshape(p.shape)
        start = end
        yield p, cur_vec


def pvec_set(obj: PObj, vec: torch.Tensor):
    """Set the params based on the vector

    Args:
        obj (PObj): The PObj to set to
        vec (torch.Tensor): The vector to set
    """

    for p, cur_vec in pvec_align(obj, vec):
        params_set(p, cur_vec)


def pvec_acc(obj: PObj, vec: torch.Tensor):
    """Accumulate the parameters

    Args:
        obj (PObj): The parameter object to accumulate
        vec (torch.Tensor): The vector to accumulate with
    """
    for p, cur_vec in pvec_align(obj, vec):
        params_acc(p, cur_vec)


def gradvec_set(obj: PObj, vec: torch.Tensor):
    """Set the gradient for a parameter object based on a vector

    Args:
        obj (PObj): The parameter object to set
        vec (torch.Tensor): The vector to set the grad with
    """

    for p, cur_vec in pvec_align(obj, vec):
        grad_set(p, cur_vec)


def gradvec_acc(obj: PObj, vec: torch.Tensor):
    """Accumulate the gradients on the parameters

    Args:
        obj (PObj): The Parameter object to accumulate for
        vec (torch.Tensor): The vector of gradients to accumulate
    """
    for p, cur_vec in pvec_align(obj, vec):
        grad_acc(p, cur_vec)


def gradtvec_set(obj: PObj, vec: torch.Tensor):
    """Set the grad vector using a target

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The target vector
    """

    for p, cur_vec in pvec_align(obj, vec):
        gradt_set(p, cur_vec)


def gradtvec_acc(obj: PObj, vec: torch.Tensor):
    """Accumulate the grad for the parameter object given a target vector

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The target vector to use
    """

    for p, cur_vec in pvec_align(obj, vec):
        gradt_acc(p, cur_vec)


def params_get(model: nn.Module) -> torch.Tensor:
    """Convenience function to retrieve the parameters of a model

    Args:
        model (nn.Module):

    Returns:
        torch.Tensor:
    """

    try:
        p = p_get(model)
        return parameters_to_vector(p)
    except NotImplementedError:
        return None


def params_to_df(name: str, obj: PObj) -> pd.DataFrame:
    """Convert parameters to a dataframe

    Args:
        name (str): The name of the column
        obj (PObj): The parameters

    Returns:
        pd.DataFrame: The parameters in dataframe form
    """
    return pd.DataFrame({name: [p for p in p_get(obj)]})


def params_to_series(obj: PObj) -> pd.DataFrame:
    """Convert the PObj to a Pandas series

    Args:
        obj (PObj): The PObj to convert

    Returns:
        pd.DataFrame: The dataframe
    """
    return pd.Series([p for p in p_get(obj)])


def multp_get(objs: typing.Iterable[PObj]) -> typing.Tuple[torch.nn.parameter.Parameter]:
    """Get params in a tuple. Primarily to
    make it easier to zip multiple modules

    Args:
        objs (typing.Iterable[PObj]): The parameter objects to get

    Returns:
        typing.Iterable[typing.Tuple[torch.nn.parameter.Parameter]]: The tuple of parameters
    """

    return tuple(p_get(obj) for obj in objs)


def p_loop(
    obj: PObj, f: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]] = None
) -> typing.Iterator[torch.nn.parameter.Parameter]:
    """Loop over the parameters for a parameter object

    Args:
        obj (PObj): The parameter object to loop over

    Yields:
        Iterator[typing.Iterator[torch.nn.parameter.Parameter]]: The parameters
    """
    for p in p_get(obj):
        if f is None:
            yield p
        else:
            yield f(p)


def p_apply(obj: PObj, f):
    """Apply a function to the parameters

    Args:
        parameters (typing.Iterator[torch.nn.parameter.Parameter]): Parameters to apply a function to
        f : The function to apply
    """
    with torch.no_grad():
        for p in p_get(obj):

            p.copy_(f(p))


def p_transfer(obj: PObj, obj2: PObj, f: typing.Callable[[torch.Tensor, torch.Tensor], typing.NoReturn]):
    """Apply a function to the parameters

    Args:
        obj1 (typing.Iterator[torch.nn.parameter.Parameter]): Parameters to apply a function to
        obj2 (typing.Iterator[torch.nn.parameter.Parameter]): Parameters to apply a function to
        f : The function to apply
    """
    with torch.no_grad():
        for p1, p2 in zip(p_get(obj), p_get(obj2)):
            f(p1, p2)


def grad_apply(obj: PObj, f: typing.Callable[[torch.Tensor, torch.Tensor], torch.Tensor], skip_none: bool = True):
    """Apply a function to the parameters

    Args:
        parameters (typing.Iterator[torch.nn.parameter.Parameter]): Parameters to apply a function to
        f : The function to apply
    """
    with torch.no_grad():
        for p in p_get(obj):
            if p.grad is None and skip_none:
                continue
            elif p.grad is None:
                p.grad = f(p, p.grad).clone()
            else:
                p.grad.copy_(f(p, p.grad))


def params_set(cur: torch.Tensor, new_: torch.Tensor):
    """Set the values of the parameters to a new value

    Args:
        cur (torch.Tensor): The current parameters to set
        new_ (torch.Tensor): The new parameters
    """
    with torch.no_grad():
        cur.copy_(new_.detach())


def params_acc(cur: torch.Tensor, dp: torch.Tensor):
    """Accumulate the parameters

    Args:
        cur (torch.Tensor): The parameters to accumulate
        dp (torch.Tensor): The change in the parameters
    """
    with torch.no_grad():
        cur.copy_(cur + dp)


def grad_set(cur: torch.Tensor, grad: torch.Tensor):
    """Set the gradient for the parameters

    Args:
        cur (torch.Tensor): The current parameters
        grad (torch.Tensor): The gradient
    """
    with torch.no_grad():
        if grad is None:
            cur.grad = None
        elif cur.grad is None:
            cur.grad = grad.clone()
        else:
            with torch.no_grad():
                cur.grad.copy_(grad.detach())


def gradt_set(cur: torch.Tensor, t: torch.Tensor):
    """Set the grad based on a target

    Args:
        cur (torch.Tensor): The current tensor
        t (torch.Tensor): The target to set as the grad
    """
    grad = cur - t
    with torch.no_grad():
        cur.grad.copy_(grad.detach())


def grad_acc(cur: torch.Tensor, grad: torch.Tensor):
    """Accumulate the gradient

    Args:
        cur (torch.Tensor): The tensor to accumulate the gradient for
        grad (torch.Tensor): The gradient to accumulate
    """
    with torch.no_grad():
        if cur.grad is None:
            cur.grad = grad.clone()
        else:
            with torch.no_grad():
                cur.grad.copy_(cur.grad + grad)


def gradt_acc(cur: torch.Tensor, t: torch.Tensor):
    """Accumualte the gradients based on a target

    Args:
        cur (torch.Tensor): The current tensor
        t (torch.Tensor): The target to use for accumulating the gradient
    """
    grad = cur - t
    with torch.no_grad():
        if cur.grad is None:
            cur.grad = grad.clone()
        else:
            with torch.no_grad():
                cur.grad.add_(grad)


def model_params_update(
    model: typing.Union[nn.Module, typing.Iterator[torch.nn.parameter.Parameter]], theta: torch.Tensor
):
    """Convenience function to update the parameters of a model

    Args:
        model (nn.Module): Model to update parameters for
        theta (torch.Tensor): The new parameters for the model
    """
    if isinstance(model, torch.nn.Module):
        model = model.parameters()
    vector_to_parameters(theta, model)


def p_reg(obj: PObj, f) -> torch.Tensor:
    """Convenience function to regularize parameters

    Args:
        obj (PObj): The parameter object ot regularize

    Returns:
        torch.Tensor: the regularization value
    """
    regularization = None
    for p in p_get(obj):

        cur = f(p)
        if cur.dim() != 0:
            raise RuntimeError("The regularization function did not output a reduced value of dim 0")
        if regularization is None:
            regularization = cur
        else:
            regularization = regularization + cur
    return regularization
