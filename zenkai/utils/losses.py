# import typing

# # 1st Party
# from abc import abstractmethod, abstractproperty
# from dataclasses import dataclass

# import numpy as np

# # 3rd Party
# import torch
# import torch.nn as nn
# import torch.nn.functional as nn_func
# from torch import nn

# # Local
# from ... import utils

# # TODO: Move as this is the wrong place
# class KLDivLoss(nn.Module):
#     """An extended version of KL divergence
#     """

#     def __init__(self, reduction: str='batchmean', class_target: bool=True, logits: bool=True):
#         super().__init__()
#         self._kl_div = nn.KLDivLoss(reduction=reduction)
#         self._inv = SigmoidInvertable()
#         self.class_target = class_target
#         self.logits = logits

#     def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#         """Calculate the KL Divergence Loss allowing class inputs

#         Args:
#             x (torch.Tensor): the input tensor
#             t (torch.Tensor): the target tensor

#         Returns:
#             Tensor: the KL divergence between the input and the output
#         """
#         if self.class_target:
#             t = torch.nn.functional.one_hot(t)

#         if self.logits:
#             x = nn_func.log_softmax(x, dim=1)
#         else:
#             x = self._inv.reverse(x)

#         return self._kl_div(x, t)


# from .assess import AssessmentDict, Reduction, Assessment


# class L2Reg(nn.Module):

#     def __init__(self, lam: float=1e-1, reduction: str='mean'):
#         super().__init__()
#         self._lam = lam
#         self._reduction = reduction
#         if reduction not in ('none', 'batchmean', 'mean'):
#             raise RuntimeError(f'Reduction must be "none", "batchmean" or "mean" {reduction} ')

#     def forward(self, x: torch.Tensor):

#         if self._reduction == 'none':
#             return (x ** 2) * self._lam

#         elif self._reduction == 'batchmean':
#             return ((x ** 2).sum() * self._lam) / len(x)

#         elif self._reduction == 'mean':
#             return (x ** 2).mean() * self._lam


# class L1Reg(nn.Module):

#     def __init__(self, lam: float=1e-1, reduction: str='mean'):
#         super().__init__()
#         self._lam = lam
#         self._reduction = reduction
#         if reduction not in ('none', 'batchmean', 'mean'):
#             raise RuntimeError(f'Reduction must be "none", "batchmean" or "mean" {reduction} ')

#     def forward(self, x: torch.Tensor):

#         # TODO: Move to Losses
#         if self._reduction == 'none':
#             return torch.abs(x) * self._lam

#         elif self._reduction == 'batchmean':
#             return (torch.abs(x) * self._lam) / len(x)

#         elif self._reduction == 'mean':
#             return torch.abs(x).mean() * self._lam


# class DXModLoss(Loss):
#     """Wraps a module and a loss together so that more advanced backpropagation
#     can be implemented
#     """

#     @abstractproperty
#     def module(self) -> nn.Module:
#         raise NotImplementedError

#     @abstractmethod
#     def forward(self, x: torch.Tensor, y: torch.Tensor, dx: torch.Tensor, reduction_override=None):
#         raise NotImplementedError

#     def assess(self, x: torch.Tensor, y: torch.Tensor, dx: torch.Tensor, reduction_override: str=None):
#         return (
#           Assessment(self.forward(x, y, dx, reduction_override), 
#           self._maximize)

#     def assess_dict(
#         self, x: torch.Tensor, y: torch.Tensor, dx: torch.Tensor, 
#         reduction_override: str=None, name: str='loss'):
#         return AssessmentDict(
#             **{name: Assessment(self.forward(x, y, dx, reduction_override), self._maximize)}
#         )


# class WeightedLoss(nn.Module):

#     def __init__(self, loss: nn.Module, weight: float=1.0, reduction='mean'):
#         super().__init__()
#         self.loss = loss
#         self.weight = weight
#         self.loss.reduction = reduction

#     def forward(self, x: torch.Tensor, t: torch.Tensor):
#         return self.weight * self.loss(x, t)

#     @property
#     def reduction(self):
#         return self.loss.reduction

#     @reduction.setter
#     def reduction(self, reduction: str):
#         self.loss.reduction = reduction


# class DXLossGrad(torch.autograd.Function):
#     """Use to implement backpropagation that is closer to the standard
#     form that sends information in a direct path backward through the layers
#     """

#     # TODO: consider making this more flexible
#     # Example: forward(*x, loss: ModLoss)
#     @staticmethod
#     def forward(ctx, x: torch.Tensor, loss: DXModLoss):
#         with torch.enable_grad():
#             x = utils.freshen(x)
#             y = loss.module.forward(x)
#         ctx.save_for_backward(x, y)
#         ctx.loss = loss
#         return y

#     @staticmethod
#     def backward(ctx, dx: torch.Tensor):

#         with torch.enable_grad():
#             x, y = ctx.saved_tensors
#             result = ctx.loss.forward(x, y, dx)
#             result.backward()

#         return x.grad, None


# class TLossGrad(torch.autograd.Function):
#     """Use to implement backpropagation that is closer to the standard
#     form that sends information in a direct path backward through the layers
#     """

#     # TODO: consider making this more flexible
#     # Example: forward(*x, loss: ModLoss)
#     @staticmethod
#     def forward(ctx, x: torch.Tensor, loss: TModLoss):
#         with torch.enable_grad():
#             x = utils.freshen(x)
#             y = loss.module.forward(x)
#         ctx.save_for_backward(x, y)
#         ctx.loss = loss
#         return y

#     @staticmethod
#     def backward(ctx, dx: torch.Tensor):

#         with torch.enable_grad():
#             x, y = ctx.saved_tensors
#             result = ctx.loss.forward(x, y, y - dx)
#             result.backward()

#         return x.grad, None
