# 1st party
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

# 3rd party
import torch
import torch.nn as nn

# local
from . import _params as param_utils
from ._params import PObj
from ._shape import feature_separate


@dataclass
class PopParams:
    """
    A class to store and manipulate parameters for a population.
    Attributes:
    -----------
    p : typing.Union[nn.parameter.Parameter, torch.Tensor]
        The parameter or tensor to be manipulated.
    n_members : int
        The number of members in the population.
    dim : int, optional
        The dimension along which to operate (default is 0).
    mixed : bool, optional
        A flag indicating if the population is mixed (default is False).
    Methods:
    --------
    pop_view():
        Returns a view of the parameter tensor based on the population settings.
    numel() -> int:
        Returns the number of elements in the parameter tensor.
    vec_reshape(vec: torch.Tensor) -> torch.Tensor:
        Reshapes a given tensor to match the shape of the parameter tensor.
    params_set(vec: torch.Tensor):
        Sets the parameters using a given tensor.
    params_acc(vec: torch.Tensor):
        Accumulates the parameters using a given tensor.
    grad_acc(vec: torch.Tensor):
        Accumulates the gradients using a given tensor.
    gradt_acc(vec: torch.Tensor):
        Accumulates the gradients (transposed) using a given tensor.
    grad_set(vec: torch.Tensor):
        Sets the gradients using a given tensor.
    gradt_set(vec: torch.Tensor):
        Sets the gradients (transposed) using a given tensor.
    """

    p: typing.Union[nn.parameter.Parameter, torch.Tensor]
    n_members: int
    dim: int = 0
    mixed: bool = False

    def pop_view(self) -> torch.Tensor:
        """
        Generate a view of the tensor population based on the specified dimensions and mixed flag.
        If the `mixed` attribute is True, it separates the features of the population tensor.
        If the `dim` attribute is not zero, it permutes the dimensions of the population tensor
        to bring the specified dimension to the front.
        Returns:
            Tensor: A view of the population tensor with the specified transformations applied.
        """

        if self.mixed:
            return feature_separate(self.p, self.n_members, self.dim, False)
        elif self.dim != 0:

            permutation = list(range(self.p.dim()))
            permutation = [
                permutation[self.dim],
                *permutation[: self.dim],
                *permutation[self.dim + 1 :],
            ]
            return self.p.permute(permutation)
        return self.p

    def numel(self) -> int:
        """
        Returns the number of elements for the parameter.
        Returns:
            int: The number of elements.
        """
        return self.p.numel()

    def vec_reshape(self, vec: torch.Tensor):

        target_shape = list(self.p.shape)

        if self.mixed:
            target_shape.insert(0, self.n_members)
            target_shape[self.dim + 1] = -1
            vec = vec.reshape(target_shape)
            vec = vec.transpose(self.dim, 0)
        elif self.dim != 0:
            target_shape[0], target_shape[self.dim] = (
                target_shape[self.dim],
                target_shape[0],
            )

            vec = vec.reshape(target_shape)
            permutation = list(range(self.p.dim()))
            permutation = [
                *permutation[: self.dim],
                0,
                *permutation[self.dim :],
            ]
            vec = vec.transpose(self.dim, 0)
        return vec.reshape_as(self.p)

    def params_set(self, vec: torch.Tensor):

        with torch.no_grad():
            vec = self.vec_reshape(vec)
            param_utils.params_set(self.p, vec.detach())

    def params_acc(self, vec: torch.Tensor):

        with torch.no_grad():
            vec = self.vec_reshape(vec)
            param_utils.params_acc(self.p, vec.detach())

    def grad_acc(self, vec: torch.Tensor):

        with torch.no_grad():
            vec = self.vec_reshape(vec)
            param_utils.grad_acc(self.p, vec.detach())

    def gradt_acc(self, vec: torch.Tensor):

        with torch.no_grad():
            vec = self.vec_reshape(vec)
            param_utils.gradt_acc(self.p, vec.detach())

    def grad_set(self, vec: torch.Tensor):

        with torch.no_grad():
            vec = self.vec_reshape(vec)
            param_utils.grad_set(self.p, vec.detach())

    def gradt_set(self, vec: torch.Tensor):

        with torch.no_grad():
            vec = self.vec_reshape(vec)
            param_utils.gradt_set(self.p, vec.detach())


class PopModule(nn.Module, ABC):
    """Parent class for a module that outputs a population"""

    def __init__(self, n_members: int, out_dim: int = 0, p_dim: int = 0, mixed: bool = False):
        """

        Args:
            n_members (int): The population size
            out_dim (int, optional): The dimension for the population for the output. Defaults to 0.
            p_dim (int, optional): The dimension for the pouplation for the parameters. Defaults to 0.
            mixed (bool, optional): Whether the population dim is mixed with another dimension. Defaults to False.
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
    def forward(self, x: torch.Tensor, ind: int = None) -> torch.Tensor:
        """Output the population

        Args:
            x (torch.Tensor): The input


        Returns:
            torch.Tensor: The population output
        """
        pass

    def pop_parameters(
        self, recurse: bool = True, pop_params: bool = True
    ) -> typing.Iterator[typing.Union[PopParams, nn.parameter.Parameter]]:

        for p in self.parameters(recurse):
            if not pop_params:
                yield feature_separate(p, self._n_members, self._p_dim, False)
            else:
                yield PopParams(p, self._n_members, self._p_dim, self._mixed)


PopM = typing.Union[typing.List[nn.Module], nn.Module]

# TODO: Loop


def to_pop_pvec(obj: PopM, n: int) -> torch.Tensor:
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
    return torch.cat(ps, dim=1)


def to_pop_gradvec(obj: PObj, n: int) -> torch.Tensor:
    """Convert the population parameters to a single tensor

    Args:
        obj (PObj): The object to get the parameters for
        n (int): The number of members

    Returns:
        torch.Tensor: The tensor representing the
    """
    return torch.cat(
        [pi_i.grad.reshape(n, -1) for pi_i in param_utils.p_get(obj)],
        dim=1,
    )


def pop_modules(m: PopModule, visited: typing.Optional[typing.Set] = None) -> typing.Iterator[nn.Module]:

    visited = visited if visited is not None else set()

    if m in visited:
        return

    visited.add(m)
    if isinstance(m, PopModule):
        yield m

    for m_i in m.children():
        for child in pop_modules(m_i, visited):
            yield child


def pop_parameters(m: PopModule, visited: typing.Optional[typing.Set] = None) -> typing.Iterator[PopParams]:

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


def pop_params_ind(
    m: PopModule, visited: typing.Optional[typing.Set] = None
) -> typing.Iterator[nn.parameter.Parameter]:

    visited = visited if visited is not None else set()
    for m_i in m.children():
        if isinstance(m_i, PopModule):
            continue
        else:
            for child in m_i.children():
                for p in pop_params_ind(child):
                    yield p
                for p in child.parameters(False):
                    yield p


def pop_vec_align(obj: PopM, vec: torch.Tensor) -> typing.Iterator[typing.Tuple[PopParams, torch.Tensor]]:
    """Align the population vector with the object passed in

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The vector to align

    Yields:
        Iterator[typing.Iterator[typing.Tuple[torch.Tensor, torch.Tensor]]]: Each parameter and aligned vector
    """
    start = 0
    for p in pop_parameters(obj):  # param_utils.p_get(obj):

        end = int(start + p.numel() / vec.shape[0])
        # Assume that the first dimension is the
        # population dimension
        cur_vec = vec[:, start:end]
        # cur_vec = cur_vec.reshape(p.shape)
        start = end
        yield p, cur_vec


def pop_vec_set(obj: PopM, vec: torch.Tensor) -> torch.Tensor:
    """Set the parameters of a PObj

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The parameter vec
    """
    for p, cur_vec in pop_vec_align(obj, vec):
        p.params_set(cur_vec)
        # param_utils.pvec_set(p, cur_vec)


def pop_vec_acc(obj: PObj, vec: torch.Tensor) -> torch.Tensor:
    """Accumulate the parameters of a PObj

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The gradient vec
    """

    for p, cur_vec in pop_vec_align(obj, vec):
        p.params_acc(cur_vec)
        # param_utils.pvec_acc(p, cur_vec)


def pop_gradvec_set(obj: PObj, vec: torch.Tensor) -> torch.Tensor:
    """Set the gradient of a PObj

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The gradient vec
    """
    for p, cur_vec in pop_vec_align(obj, vec):
        p.grad_set(cur_vec)
        # param_utils.grad_set(p, cur_vec)


def pop_gradvec_acc(obj: PObj, vec: torch.Tensor) -> torch.Tensor:
    """Accumulate the gradient of a PObj

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The gradient vec
    """
    for p, cur_vec in pop_vec_align(obj, vec):
        p.grad_acc(cur_vec)
        # p.grad_acc(cur_vec)
        # param_utils.grad_acc(p, cur_vec)


def pop_gradtvec_set(obj: PObj, vec: torch.Tensor) -> torch.Tensor:
    """Set the gradient of a PObj based on a target vector

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The target vec
    """
    for p, cur_vec in pop_vec_align(obj, vec):
        p.gradt_set(cur_vec)
        # param_utils.gradt_set(p, cur_vec)


def pop_gradtvec_acc(obj: PObj, vec: torch.Tensor) -> torch.Tensor:
    """Acc the gradient of a PObj based on a target vector

    Args:
        obj (PObj): The parameter object
        vec (torch.Tensor): The target vec
    """
    for p, cur_vec in pop_vec_align(obj, vec):
        p.gradt_acc(cur_vec)
        # param_utils.gradt_acc(p, cur_vec)
