# 1st party
from abc import abstractmethod, ABC
import typing
from dataclasses import dataclass

# 3rd party
import torch
import torch.nn as nn

# local
from . import _params
from ..utils import _params as base_params
from ._reshape import collapse_batch, collapse_feature, separate_batch, separate_feature
from ..utils import _params as param_utils



@dataclass
class PopParams:

    p: typing.Union[nn.parameter.Parameter, torch.Tensor]
    n_members: int
    dim: int=0
    mixed: bool=False

    def pop_view(self):
        if self.mixed:
            return separate_feature(
                self.p, self.n_members, self.dim, False
            )
        elif self.dim != 0:
    
            permutation = list(range(self.p.dim()))
            permutation = [
                permutation[self.dim], 
                *permutation[:self.dim],
                *permutation[self.dim + 1:]
            ]
            return self.p.permute(permutation)
        return self.p

    def numel(self) -> int:
        return self.p.numel()

    def reshape_vec(self, vec: torch.Tensor):

        target_shape = list(self.p.shape)

        if self.mixed:
            target_shape.insert(0, self.n_members)
            target_shape[self.dim + 1] = -1
            vec = vec.reshape(target_shape)
            vec = vec.transpose(self.dim, 0)
        elif self.dim != 0:
            target_shape[0], target_shape[self.dim] = target_shape[self.dim], target_shape[0]

            vec = vec.reshape(target_shape)
            permutation = list(range(self.p.dim()))
            permutation = [ 
                *permutation[:self.dim],
                0,
                *permutation[self.dim:]
            ]
            vec = vec.transpose(self.dim, 0)
        return vec.reshape_as(self.p)
    
    def set_params(self, vec: torch.Tensor):

        with torch.no_grad():
            vec = self.reshape_vec(vec)
            param_utils.set_params(
                self.p, vec.detach()
            )

    def acc_params(self, vec: torch.Tensor):

        with torch.no_grad():
            vec = self.reshape_vec(vec)
            param_utils.acc_params(
                self.p, vec.detach()
            )

    def acc_grad(self, vec: torch.Tensor):

        with torch.no_grad():
            vec = self.reshape_vec(vec)
            param_utils.acc_grad(
                self.p, vec.detach()
            )

    def acc_gradt(self, vec: torch.Tensor):

        with torch.no_grad():
            vec = self.reshape_vec(vec)
            param_utils.acc_gradt(
                self.p, vec.detach()
            )
    def set_grad(self, vec: torch.Tensor):

        with torch.no_grad():
            vec = self.reshape_vec(vec)
            param_utils.set_grad(
                self.p, vec.detach()
            )
    
    def set_gradt(self, vec: torch.Tensor):

        with torch.no_grad():
            vec = self.reshape_vec(vec)
            param_utils.set_gradt(
                self.p, vec.detach()
            )


class PopModule(nn.Module, ABC):
    """Parent class for a module that outputs a population
    """
    def __init__(
        self, n_members: int, out_dim: int=0, p_dim: int=0, mixed: bool=False):
        """Create a population module with the specified number of members

        Args:
            n_members (int): The number of members in the population
        """
        super().__init__()
        self._n_members = n_members
        self._out_dim = out_dim
        self._p_dim = p_dim
        self._mixed = mixed

    @property
    def n_members(self) -> int:
        """
        Returns:
            int: The number of members in the module
        """
        return self._n_members

    @abstractmethod
    def forward(
        self, x: torch.Tensor, 
        ind: int=None
    ) -> torch.Tensor:
        """Output the population

        Args:
            x (torch.Tensor): The input


        Returns:
            torch.Tensor: The population output
        """
        pass 

    def pop_parameters(self, recurse: bool=True, pop_params: bool=True) -> typing.Iterator[typing.Union[PopParams, nn.parameter.Parameter]]:

        for p in self.parameters(recurse):
            if not pop_params:
                yield separate_feature(
                    p, self._n_members, self._p_dim, False
                )
            else:
                yield PopParams(
                    p, self._n_members, self._p_dim, self._mixed
                )


def chained(*mods: nn.Module) -> nn.ModuleList:
    return nn.ModuleList(mods)


# TODO: add tests
class AdaptBatch(nn.Module):
    """Use to adapt a population of samples for evaluating perturbations
    of samples. Useful for optimizing "x"

    """

    def __init__(self, module: nn.Module):
        """Instantiate the AdaptBatch model

        Args:
            module (nn.Module): 
        """
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt the module with  apopulation separating by the batch dimension

        Returns:
            torch.Tensor: The output of the module
        """
        k = x[0].size(0)
        
        x = tuple(collapse_batch(x_i) for x_i in x)
        x = self.module(*x)
        if isinstance(x, typing.Tuple):
            return tuple(
                separate_batch(x_i, k) for x_i in x
            )
        
        return separate_batch(x, k)


class AdaptFeature(PopModule):
    """Use to adapt a population of samples for evaluating perturbations of models. 
    """

    def __init__(self, module: nn.Module, n_members: int=None, feature_dim: int=1):
        """Adapt module to work with a population of inputs

        Args:
            module (nn.Module): The module to a adapt
        """
        super().__init__(n_members, feature_dim, True)
        self.module = module
        self.feature_dim = feature_dim

    def forward(self, *x: torch.Tensor) -> torch.Tensor:
        """Adapt the module with  apopulation separating by feature

        Returns:
            torch.Tensor: The output of the module
        """
        k = x[0].size(0)
        
        x = tuple(collapse_feature(x_i, self.feature_dim) for x_i in x)
        x = self.module(*x)

        if isinstance(x, typing.Tuple):
            return tuple(
                separate_feature(x_i, k, self.feature_dim) for x_i in x
            )
        return separate_feature(x, k, self.feature_dim)
