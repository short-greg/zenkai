# 1st party
import typing

# 3rd party
import torch
import torch.nn as nn

# local
from ..utils import _params as param_utils
from ..utils._params import PObj
from ._selection import Selection
from ._module import PopModule, PopParams


PopM = typing.Union[typing.List[nn.Module], nn.Module]


def loop_select(
    obj: PObj, selection: Selection
) -> typing.Iterator[typing.Tuple[torch.Tensor, torch.Tensor]]:
    """Loop over a parameter object and call a function

    Args:
        obj (PObj): The parameter object
        selection (Selection): The selection for the parameter object
        f (typing.Callable): The function to execute

    Yields:
        typing.Tuple[torch.Tensor, torch.Tensor]: The selected parameter and assessment
    """
    for p in param_utils.get_p(obj):

        selected, assessment_i = selection(
            p, get_assessment=True
        )
        # assessment_i = tansaku_utils.unsqueeze_to(
        #     selection.assessment, selected
        # )
        # if f is not None:
        #     res = f(selected, assessment_i, selection.n, selection.k)
        yield selected, assessment_i


def to_pvec(obj: PopM, n: int) -> torch.Tensor:
    """Convert the population parameters to a single tensor

    # Note: Assumes the population dimension is 0
    # for all
    Args:
        obj (PObj): The object to get the parameters for
        n (int): The number of members

    Returns:
        torch.Tensor: The tensor representing the 
    """
    ps = [pi_i.pop_view().reshape(n, -1) for pi_i in pop_parameters(obj)]
    if len(ps) == 0:
        return None
    return torch.cat(
        ps, dim=1
    )


def to_gradvec(obj: PObj, n: int) -> torch.Tensor:
    """Convert the population parameters to a single tensor

    Args:
        obj (PObj): The object to get the parameters for
        n (int): The number of members

    Returns:
        torch.Tensor: The tensor representing the 
    """
    return torch.cat(
        [pi_i.grad.reshape(n, -1) for pi_i in param_utils.get_p(obj)], 
        dim=1
    )


def pop_modules(m: PopModule, visited: typing.Optional[typing.Set]=None) -> typing.Iterator[nn.Module]:

    visited = visited if visited is not None else set()

    if m in visited:
        return
    
    visited.add(m)
    if isinstance(m, PopModule):
        yield m

    for m_i in m.children():
        for child in pop_modules(m_i, visited):
            yield child


def pop_parameters(m: PopModule, visited: typing.Optional [typing.Set]=None) -> typing.Iterator[PopParams]:

    visited = visited if visited is not None else set()

    if m in visited:
        return

    visited.add(m)
    if isinstance(m, PopModule):
        for p in m.pop_parameters():
            yield p

    for m_i in m.children():
        for p in pop_parameters(m_i, visited):
            yield p


def ind_parameters(m: PopModule, visited: typing.Optional [typing.Set]=None) -> typing.Iterator[nn.parameter.Parameter]:

    visited = visited if visited is not None else set()
    for m_i in m.children():
        if isinstance(m_i, PopModule):
            continue
        else:
            for child in m_i.children():
                for p in ind_parameters(child):
                    yield p
                for p in child.parameters(False):
                    yield p


def align_vec(obj: PopM, vec: torch.Tensor) -> typing.Iterator[typing.Tuple[PopParams, torch.Tensor]]:
    """Align the population vector with the object passed in

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The vector to align

    Yields:
        Iterator[typing.Iterator[typing.Tuple[torch.Tensor, torch.Tensor]]]: Each parameter and aligned vector
    """

    start = 0
    for p in pop_parameters(obj): # param_utils.get_p(obj):

        end = int(start + p.numel() / vec.shape[0])
        # Assume that the first dimension is the
        # population dimension
        cur_vec = vec[:,start:end]
        # cur_vec = cur_vec.reshape(p.shape)
        start = end
        yield p, cur_vec


def set_pvec(obj: PopM, vec: torch.Tensor) -> torch.Tensor:
    """Set the parameters of a PObj

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The gradient vec
    """
    for p, cur_vec in align_vec(obj, vec):
        p.set_params(cur_vec)
        # param_utils.set_pvec(p, cur_vec)


def acc_pvec(obj: PObj, vec: torch.Tensor) -> torch.Tensor:
    """Accumulate the parameters of a PObj

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The gradient vec
    """

    for p, cur_vec in align_vec(obj, vec):
        p.acc_params(cur_vec)
        # param_utils.acc_pvec(p, cur_vec)


def set_gradvec(obj: PObj, vec: torch.Tensor) -> torch.Tensor:
    """Set the gradient of a PObj

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The gradient vec
    """
    for p, cur_vec in align_vec(obj, vec):
        p.set_grad(cur_vec)
        # param_utils.set_grad(p, cur_vec)


def acc_gradvec(obj: PObj, vec: torch.Tensor) -> torch.Tensor:
    """Accumulate the gradient of a PObj

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The gradient vec
    """
    for p, cur_vec in align_vec(obj, vec):
        p.acc_grad(cur_vec)
        # p.acc_grad(cur_vec)
        # param_utils.acc_grad(p, cur_vec)


def set_gradtvec(obj: PObj, vec: torch.Tensor) -> torch.Tensor:
    """Set the gradient of a PObj based on a target vector

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The target vec
    """
    for p, cur_vec in align_vec(obj, vec):
        p.set_gradt(cur_vec)
        # param_utils.set_gradt(p, cur_vec)


def acc_gradtvec(obj: PObj, vec: torch.Tensor) -> torch.Tensor:
    """Acc the gradient of a PObj based on a target vector

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The target vec
    """
    for p, cur_vec in align_vec(obj, vec):
        p.acc_gradt(cur_vec)
        # param_utils.acc_gradt(p, cur_vec)
