
# 1st Party
import typing

# 3rd party
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from itertools import chain
import pandas as pd


PObj = typing.Union[nn.Module, typing.Iterator[torch.nn.parameter.Parameter], torch.Tensor, typing.Callable[[],typing.Iterator[torch.nn.parameter.Parameter]]]


def get_p(obj: PObj) -> typing.Iterable[torch.nn.parameter.Parameter]:
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


def get_grad(obj: PObj) -> typing.Iterator[torch.nn.parameter.Parameter]:
    """Get the gradients for a PObj

    Args:
        obj (PObj): The PObj to get parameters for

    yields:
        torch.nn.parameter.Parameter: All the gradients for tetheh PObj 
    """
    
    for p in get_p(obj):
        yield p.grad


def to_pvec(obj: PObj) -> torch.Tensor:
    """Convert a PObj to a flattened vector

    Args:
        obj (PObj): The object to convert

    Returns:
        torch.Tensor: The vector
    """

    return torch.cat([p_i.flatten() for p_i in get_p(obj)], dim=0)


def to_gradvec(obj: PObj) -> torch.Tensor:
    """Retrieve a vector of gradients from the grad object

    Args:
        obj (PObj): The object to convert

    Returns:
        torch.Tensor: The Gradient 
    """

    result = []
    for p_i in get_p(obj):
        if p_i.grad is None:
            result.append(torch.zeros_like(p_i).flatten())
        else:
            result.append(p_i.grad.flatten())
    return torch.cat(result, dim=0)


def align_pvec(obj: PObj, vec: torch.Tensor) -> typing.Iterator[typing.Tuple[torch.Tensor, torch.Tensor]]:
    """Align a vector to parmaters for a PObject

    Args:
        obj (PObj): The PObject to align to
        vec (torch.Tensor): The vector to align


    Yields:
        Iterator[typing.Iterator[typing.Tuple[torch.Tensor, torch.Tensor]]]: The aligned vectors
    """
    start = 0
    for p in get_p(obj):

        end = start + p.numel()
        cur_vec = vec[start:end]
        cur_vec = cur_vec.reshape(p.shape)
        start = end
        yield p, cur_vec


def set_pvec(obj: PObj, vec: torch.Tensor):
    """Set the params based on the vector

    Args:
        obj (PObj): The PObj to set to
        vec (torch.Tensor): The vector to set
    """

    for p, cur_vec in align_pvec(obj, vec):
        set_params(p, cur_vec)


def acc_pvec(obj: PObj, vec: torch.Tensor):
    """Accumulate the parameters

    Args:
        obj (PObj): The parameter object to accumulate
        vec (torch.Tensor): The vector to accumulate with
    """
    for p, cur_vec in align_pvec(obj, vec):
        acc_params(p, cur_vec)


def set_gradvec(obj: PObj, vec: torch.Tensor):
    """Set the gradient for a parameter object based on a vector

    Args:
        obj (PObj): The parameter object to set
        vec (torch.Tensor): The vector to set the grad with
    """
    
    for p, cur_vec in align_pvec(obj, vec):
        set_grad(p, cur_vec)


def acc_gradvec(obj: PObj, vec: torch.Tensor):
    """Accumulate the gradients on the parameters

    Args:
        obj (PObj): The Parameter object to accumulate for
        vec (torch.Tensor): The vector of gradients to accumulate
    """
    for p, cur_vec in align_pvec(obj, vec):
        acc_grad(p, cur_vec)


def set_gradtvec(obj: PObj, vec: torch.Tensor):
    """Set the grad vector using a target

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The target vector
    """
    
    for p, cur_vec in align_pvec(obj, vec):
        set_gradt(p, cur_vec)


def acc_gradtvec(obj: PObj, vec: torch.Tensor):
    """Accumulate the grad for the parameter object given a target vector
    
    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The target vector to use 
    """

    for p, cur_vec in align_pvec(obj, vec):
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


def params_to_df(name: str, obj: PObj) -> pd.DataFrame:
    """Convert parameters to a dataframe

    Args:
        name (str): The name of the column
        obj (PObj): The parameters

    Returns:
        pd.DataFrame: The parameters in dataframe form
    """
    return pd.DataFrame(
        {name: [p for p in get_p(obj)]}
    )


def params_to_series(obj: PObj) -> pd.DataFrame:
    """Convert the PObj to a Pandas series

    Args:
        obj (PObj): The PObj to convert

    Returns:
        pd.DataFrame: The dataframe
    """
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


def loop_p(obj: PObj, f: typing.Optional[typing.Callable[[torch.Tensor], torch.Tensor]]=None) -> typing.Iterator[torch.nn.parameter.Parameter]:
    """Loop over the parameters for a parameter object

    Args:
        obj (PObj): The parameter object to loop over

    Yields:
        Iterator[typing.Iterator[torch.nn.parameter.Parameter]]: The parameters
    """
    for p in get_p(obj):
        if f is None:
            yield p
        else:
            yield f(p)


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


def transfer_p(
    obj: PObj, obj2: PObj, f: typing.Callable[[torch.Tensor, torch.Tensor], typing.NoReturn]
):
    """Apply a function to the parameters

    Args:
        obj1 (typing.Iterator[torch.nn.parameter.Parameter]): Parameters to apply a function to
        obj2 (typing.Iterator[torch.nn.parameter.Parameter]): Parameters to apply a function to
        f : The function to apply
    """
    with torch.no_grad():
        for p1, p2 in zip(get_p(obj), get_p(obj2)):
            f(p1, p2)


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
    """Set the values of the parameters to a new value

    Args:
        cur (torch.Tensor): The current parameters to set
        new_ (torch.Tensor): The new parameters
    """
    with torch.no_grad():
        cur.copy_(new_.detach())


def acc_params(
    cur: torch.Tensor, dp: torch.Tensor
):
    """Accumulate the parameters
    
    Args:
        cur (torch.Tensor): The parameters to accumulate
        dp (torch.Tensor): The change in the parameters
    """
    with torch.no_grad():
        cur.copy_(cur + dp)


def set_grad(
    cur: torch.Tensor, grad: torch.Tensor
):
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
    

def set_gradt(
    cur: torch.Tensor, t: torch.Tensor
):
    """Set the grad based on a target

    Args:
        cur (torch.Tensor): The current tensor
        t (torch.Tensor): The target to set as the grad
    """
    grad = cur - t
    with torch.no_grad():
        cur.grad.copy_(grad.detach())


def acc_grad(
    cur: torch.Tensor, grad: torch.Tensor
):
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


def acc_gradt(
    cur: torch.Tensor, t: torch.Tensor
):
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
    return regularization


class undo_grad(object):
    """Context that allows the user to run an operation that updates the parameters and then sets them back for a subset of those parameters.
    """

    def __init__(
        self, values: typing.Iterable[typing.Union[typing.Callable[[], typing.Iterator], nn.Module, torch.Tensor, nn.parameter.Parameter]]
    ):
        """Undoes updates to the gradients on the parameters passed in. Useful if you only want to update a subset of the gradients

        Args:
            values (typing.Iterable[typing.Union[typing.Callable[[], typing.Iterator], nn.Module, torch.Tensor, nn.parameter.Parameter]]): The values
        """
        if isinstance(values, nn.Module) or isinstance(values, torch.Tensor):
            values = [values]
        
        self._values = values        
        self._stored = []
        for value in self._values:
            if (
                isinstance(value, torch.Tensor) or isinstance(value, torch.nn.parameter.Parameter)
            ):
                self._stored.append(
                    value.grad.clone() if value.grad is not None else None
                )
            else:
                self._stored.append(
                    list(loop_p(value, lambda v: v.grad.clone() if v.grad is not None else None))
                    # get_model_grads(value)
                )

    def __enter__(self):

        pass
        
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
                transfer_p(
                    value, stored, lambda p1, p2: set_grad(
                        p1, p2
                    )
                )
