# 3rd party
import torch
import torch.nn as nn
import torch.nn.functional


class Lambda(nn.Module):
    """
    Use as a
    """

    def __init__(self, f):
        super().__init__()
        self._f = f

    def forward(self, *x: torch.Tensor):
        return self._f(*x)


class Argmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self._dim = dim

    def forward(self, x: torch.Tensor):
        return torch.argmax(x, dim=-1)


class Sign(nn.Module):
    def __init__(self, grad: bool = True):
        super().__init__()
        self._grad = grad

    def forward(self, x: torch.Tensor):
        if self._grad:
            return SignFG.apply(x)
        return torch.sign(x)


class SignFG(torch.autograd.Function):
    """Use to clip the grad between two values
    Useful for smooth maximum/smooth minimum
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return torch.sign(x).type_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad


# ##### Least squares
# class LinearLayer(nn.Module):

#     def __init__(
#       self, in_features: int, out_features: int, bias: bool=True, 
#       in_activation: nn.Module=None, out_activation: nn.Module=None):

#         super().__init__()
#         self._in_features = in_features
#         self._out_features = out_features
#         self.linear = nn.Linear(in_features, out_features, bias=bias)
#         self.in_activation = in_activation
#         self.out_activation = out_activation

#     @property
#     def weight(self) -> nn.parameter.Parameter:
#         return self.linear.weight

#     @property
#     def bias(self) -> nn.parameter.Parameter:
#         return self.linear.bias

#     @property
#     def use_bias(self) -> bool:
#         return self.linear.bias is not None

#     def sub(self, idx: torch.LongTensor):
#         layer = LinearLayer(
#             self._in_features, len(idx), self.linear.bias is not None,
#             self.in_activation, self.out_activation
#         )
#         layer.linear.weight = nn.parameter.Parameter(self.linear.weight[idx])
#         layer.linear.bias = nn.parameter.Parameter(self.linear.bias[idx])
#         return layer

#     @property
#     def weight_with_bias(self) -> torch.Tensor:
#         if self.bias is not None:
#             return torch.cat(
#                 [self.weight, self.bias[:,None]], dim=1
#             )
#         return self.weight

#     def forward(self, x: torch.Tensor):
#         if self.in_activation is not None:
#             x = self.in_activation(x)
#         x = self.linear(x)
#         if self.out_activation is not None:
#             x = self.out_activation(x)
#         return x


# class BatchNorm1DS(nn.BatchNorm1d):

#     def forward(self, x: torch.Tensor, update_stats: bool=True):

#         if update_stats or not self.training:
#             normalized = super().forward(x)
#             print(normalized.mean(dim=0), normalized.std(dim=0))
#             return normalized

#         normalized = (x - x.mean(dim=0, keepdim=True)) / torch.sqrt(x.var(dim=0, keepdim=True) + self.eps)
#         if self.affine:
#             return normalized  * self.weight[None] + self.bias[None]
#         return normalized


# Rename
# class KLDivLossWithLogits(nn.Module):

#     def __init__(self, reduction: str='batchmean', class_target: bool=True):
#         super().__init__()
#         self._kl_div = nn.KLDivLoss(reduction=reduction)
#         self._class_target = class_target

#     def forward(self, x: torch.Tensor, t: torch.Tensor):
#         if self._class_target:
#             t = torch.nn.functional.one_hot(t)
#         return self._kl_div(nn_func.log_softmax(x, dim=1), t)


# # TODO: Remove if it does not work
# class BinaryStepAdv(torch.autograd.Function):
#     """Use to clip the grad between two values
#     Useful for smooth maximum/smooth minimum
#     """

#     @staticmethod
#     def forward(ctx, x):
#         """
#         Forward pass of the Binary Step function.
#         """
#         ctx.save_for_backward(x)
#         return torch.sign(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Backward pass of the Binary Step function using the Straight-Through Estimator.
#         """
#         x, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         query = (
#             ((x < 1) & (grad_output > 0)) | ((x > 1) & (grad_output < 0))
#         )
#         grad_input[query] = 0
#         return grad_input
