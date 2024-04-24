# 1st party
import typing

# 3rd party
import torch

# local
from ..utils import _params as param_utils
from ..utils._params import PObj
from ._selection import Selection


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


def to_pvec(obj: PObj, n: int) -> torch.Tensor:
    """Convert the population parameters to a single tensor

    Args:
        obj (PObj): The object to get the parameters for
        n (int): The number of members

    Returns:
        torch.Tensor: The tensor representing the 
    """
    return torch.cat(
        [pi_i.reshape(n, -1) for pi_i in param_utils.get_p(obj)], 
        dim=0
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
        dim=0
    )


def align_vec(obj: PObj, vec: torch.Tensor) -> typing.Iterator[typing.Tuple[torch.Tensor, torch.Tensor]]:
    """Align the population vector with the object passed in

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The vector to align

    Yields:
        Iterator[typing.Iterator[typing.Tuple[torch.Tensor, torch.Tensor]]]: Each parameter and aligned vector
    """
    start = 0
    for p in param_utils.get_p(obj):

        end = start + p[0].numel()
        # Assume that the first dimension is the
        # population dimension
        cur_vec = vec[:,start:end]
        cur_vec = cur_vec.reshape(p.shape)
        start = end
        yield p, cur_vec


def set_pvec(obj: PObj, vec: torch.Tensor) -> torch.Tensor:
    """Set the parameters of a PObj

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The gradient vec
    """
    for p, cur_vec in align_vec(obj, vec):
        param_utils.set_pvec(p, cur_vec)


def acc_pvec(obj: PObj, vec: torch.Tensor) -> torch.Tensor:
    """Accumulate the parameters of a PObj

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The gradient vec
    """
    for p, cur_vec in align_vec(obj, vec):
        param_utils.acc_pvec(p, cur_vec)


def set_gradvec(obj: PObj, vec: torch.Tensor) -> torch.Tensor:
    """Set the gradient of a PObj

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The gradient vec
    """
    for p, cur_vec in align_vec(obj, vec):
        param_utils.set_grad(p, cur_vec)


def acc_gradvec(obj: PObj, vec: torch.Tensor) -> torch.Tensor:
    """Accumulate the gradient of a PObj

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The gradient vec
    """
    for p, cur_vec in align_vec(obj, vec):
        param_utils.acc_grad(p, cur_vec)


def set_gradtvec(obj: PObj, vec: torch.Tensor) -> torch.Tensor:
    """Set the gradient of a PObj based on a target vector

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The target vec
    """
    for p, cur_vec in align_vec(obj, vec):
        param_utils.set_gradt(p, cur_vec)


def acc_gradtvec(obj: PObj, vec: torch.Tensor) -> torch.Tensor:
    """Acc the gradient of a PObj based on a target vector

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The target vec
    """
    for p, cur_vec in align_vec(obj, vec):
        param_utils.acc_gradt(p, cur_vec)
