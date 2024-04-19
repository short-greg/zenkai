# 1st party
import typing

# 3rd party
import torch

# local
from .. import utils
from ..utils import PObj
from . import _reshape as tansaku_utils
from ._selection import Selection


def loop_p(
    obj: utils.PObj, selection: Selection, f: typing.Callable
):
    for p in utils.get_p(obj):

        selected = selection(p)
        assessment_i = tansaku_utils.unsqueeze_to(
            selection.assessment, selected
        )
        res = f(selected, assessment_i, selection.n, selection.k)
        yield res


def to_pvec(obj: utils.PObj, n: int) -> torch.Tensor:

    return torch.cat(
        [pi_i.reshape(n, -1) for pi_i in utils.get_p(obj)], 
        dim=0
    )


def align_vec(obj: PObj, vec: torch.Tensor) -> typing.Iterator[typing.Tuple[torch.Tensor, torch.Tensor]]:
    start = 0
    for p in utils.get_p(obj):

        end = start + p.numel()
        cur_vec = vec[:,start:end]
        cur_vec = cur_vec.reshape(p)
        yield p, cur_vec


def set_gradvec(obj: PObj, vec: torch.Tensor) -> torch.Tensor:
    
    for p, cur_vec in align_vec(obj, vec):
        utils.set_grad(p, cur_vec)


def acc_gradvec(obj: PObj, vec: torch.Tensor) -> torch.Tensor:

    for p, cur_vec in align_vec(obj, vec):
        utils.acc_grad(p, cur_vec)


def set_gradtvec(obj: PObj, vec: torch.Tensor) -> torch.Tensor:
    
    for p, cur_vec in align_vec(obj, vec):
        utils.set_gradt(p, cur_vec)


def acc_gradtvec(obj: PObj, vec: torch.Tensor) -> torch.Tensor:

    for p, cur_vec in align_vec(obj, vec):
        utils.acc_gradt(p, cur_vec)


def apply_p(
    obj: utils.PObj, selection: Selection, f: typing.Callable
):
    for _ in loop_p(
        obj, selection, f
    ):
        pass


# TODO: Review this

# def loop_multip(
#     obj: typing.Iterable[utils.PObj], selection: Selection, f: typing.Callable,
#     others: typing.Iterable
# ):
#     for cur in zip(utils.get_multip(obj), *others):
#         ps = cur[0]
#         others = cur[1:]

#         selected = selection.multi(ps)
#         assessment_i = tansaku_utils.unsqueeze_to(
#             selection.assessment, selected
#         )
#         res = f(selected, assessment_i)
#         yield res



# ps, ... in zip(utils.get_multip(...), ...)
#    selection.multi(ps)

# def loop_multip(
#     obj: utils.PObj, selection: Selection, f: typing.Callable,
#     others
# ):
#     for p in utils.get_multip(obj):

#         selected = selection(p)
#         assessment_i = tansaku_utils.unsqueeze_to(
#             selection.assessment, selected
#         )
#         res = f(selected, assessment_i, selection.n, selection.k)
#         yield res



# loop_p(obj))

# def set_p(obj: utils.PObj, selection: Selection, f: typing.Callable):
#     """Use a function to update each pop parameter

#     Args:
#         net (utils.PObj): The net to update
#         selection (Selection): The selection to use
#         f (typing.Callable): The function to call to update
#         update_grad (bool, optional): Whether to set the grad of the parameter rather than parameter itself. Defaults to False.
#     """

#     for p, new_p in loop_p(obj, selection, f):

#         assert p.data.shape == new_p.shape, new_p.shape
#         p.data = new_p.detach()


# def acc_p(net: utils.PObj, selection: Selection, f: typing.Callable):
#     """Use a function to update each pop parameter

#     Args:
#         net (utils.PObj): The net to update
#         selection (Selection): The selection to use
#         f (typing.Callable): The function to call to update
#         update_grad (bool, optional): Whether to set the grad of the parameter rather than parameter itself. Defaults to False.
#     """

#     for p, new_p in utils.get_p(net, selection, f):

#         assert p.data.shape == new_p.shape, new_p.shape
#         p.data = (p.data + new_p).detach()


# def acc_g(obj: utils.PObj, selection: Selection, f: typing.Callable):
#     """Use a function to update each pop parameter

#     Args:
#         net (utils.PObj): The net to update
#         selection (Selection): The selection to use
#         f (typing.Callable): The function to call to update
#         update_grad (bool, optional): Whether to set the grad of the parameter rather than parameter itself. Defaults to False.
#     """

#     for p, grad in utils.get_p(obj, selection, f):

#         # diff = p - new_p
#         if p.grad is None:
#             p.grad = grad.detach()
#         else:
#             assert p.grad.shape == grad.shape, grad.shape
#             p.grad.data = (p.grad + grad).detach()

# def acc_gt(obj: utils.PObj, selection: Selection, f: typing.Callable):
#     """Use a function to update each pop parameter

#     Args:
#         net (utils.PObj): The net to update
#         selection (Selection): The selection to use
#         f (typing.Callable): The function to call to update
#         update_grad (bool, optional): Whether to set the grad of the parameter rather than parameter itself. Defaults to False.
#     """

#     for p, grad in utils.get_p(obj, selection, f):

#         # diff = p - new_p
#         if p.grad is None:
#             p.grad = grad.detach()
#         else:
#             assert p.grad.shape == grad.shape, grad.shape
#             p.grad.data = (p.grad + grad).detach()


# def set_g(obj: utils.PObj, selection: Selection, f: typing.Callable):
#     """Use a function to update each pop parameter

#     Args:
#         net (utils.PObj): The net to update
#         selection (Selection): The selection to use
#         f (typing.Callable): The function to call to update
#         update_grad (bool, optional): Whether to set the grad of the parameter rather than parameter itself. Defaults to False.
#     """

#     for p, grad in utils.get_p(obj, selection, f):

#         # selected = selection(p)
#         # assessment_i = tansaku_utils.unsqueeze_to(
#         #     selection.assessment, selected)
#         # new_p = f(selected, assessment_i)
#         # diff = p - new_p
#         if p.grad is None:
#             p.grad = grad.detach()
#         else:
#             assert p.grad.shape == grad.shape, grad.shape
#             p.grad.data = grad.detach()


# TODO: Add module wrapper to do "forward_pop"
# 
