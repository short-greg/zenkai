# 1st party
import typing
import torch.nn as nn

# 3rd party
import torch

# local
from .. import utils
from . import _utils as tansaku_utils
from ._selection import Selection


def update_pop_params(net: utils.PObj, selection: Selection, f: typing.Callable):
    """Use a function to update each pop parameter

    Args:
        net (utils.PObj): The net to update
        selection (Selection): The selection to use
        f (typing.Callable): The function to call to update
        update_grad (bool, optional): Whether to set the grad of the parameter rather than parameter itself. Defaults to False.
    """

    for p in utils.get_params(net):

        selected = selection(p)
        assessment_i = tansaku_utils.unsqueeze_to(selection.assessment, selected)
        new_p = f(selected, assessment_i)
        p.data = new_p.detach()


def update_pop_grads(net: utils.PObj, selection: Selection, f: typing.Callable):
    """Use a function to update each pop parameter

    Args:
        net (utils.PObj): The net to update
        selection (Selection): The selection to use
        f (typing.Callable): The function to call to update
        update_grad (bool, optional): Whether to set the grad of the parameter rather than parameter itself. Defaults to False.
    """

    for p in utils.get_params(net):

        selected = selection(p)
        assessment_i = tansaku_utils.unsqueeze_to(selection.assessment, selected)
        new_p = f(selected, assessment_i)
        diff = p - new_p
        if p.grad is None:
            p.grad = diff.detach()
        else:
            p.grad.data += diff.detach()


# TODO: Add module wrapper to do "forward_pop"
# 
