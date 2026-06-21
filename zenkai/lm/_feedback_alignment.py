# 1st party
import typing

# 3rd party
import torch
import torch.nn as nn
from torch._tensor import Tensor

# local
from zenkai._core import IO, State
from zenkai._core import _params as param_utils
from zenkai._core import iou
from zenkai.nnz import Criterion, NNLoss, Null
from zenkai.optimz import OptimFactory

from ._grad import GradLearner
from ._lm import OutT, forward_dep


def fa_target(y: IO, y_prime: IO, detach: bool = True) -> IO:
    """Create the target for feedback alignment.

    Args:
        y (IO): The original output of the layer.
        y_prime (IO): The updated target.
        detach (bool, optional): Whether to detach. Defaults to True.

    Returns:
        IO: The resulting target.
    """
    return iou(y.f, y_prime.f, detach=detach)


class FALearner(GradLearner):
    """Learner for implementing feedback alignment."""

    def __init__(
        self,
        net: nn.Module,
        netB: nn.Module,
        activation: nn.Module = None,
        optim_factory: OptimFactory = None,
        learn_criterion: typing.Union[Criterion, str] = "MSELoss",
    ) -> None:
        """Wrap a module to create an FALearner.

        It is flexible but somewhat computationally wasteful because it executes forward on netB.

        Args:
            net (nn.Module): The net to use for forward prop.
            netB (nn.Module): The net to use for backprop.
            activation (nn.Module): The activation. Defaults to None.
            optim_factory (OptimFactory): The optimizer. Defaults to None.
            learn_criterion (typing.Union[Criterion, str], optional): The criterion. Defaults to "MSELoss".
        """
        if isinstance(learn_criterion, str):
            learn_criterion = NNLoss(learn_criterion)

        super().__init__(module=net, criterion=learn_criterion)
        self.net = net
        self.netB = netB
        self.activation = activation or Null()
        self.flatten = nn.Flatten()
        self._optim = optim_factory(self.net.parameters())

    def forward_nn(self, x: IO, state: State) -> torch.Tensor:
        """Pass the input through the net.

        Args:
            x (IO): The input.
            state (State): The learning state.

        Returns:
            torch.Tensor: The output of the net after activation.
        """
        y = self.net(x.f)
        y = y.detach()
        state._y_det = y
        y.requires_grad = True
        y.retain_grad()
        return self.activation(y)

    def accumulate(self, x: IO, t: IO, state: State):
        """Accumulate the gradients.

        Args:
            x (IO): The input.
            t (IO): The target.
            state (State): The learning state.
        """
        y = state._y
        y2 = self.netB(x.f)

        self.criterion.assess(y, t).backward()
        y_det = state._y_det
        y2.backward(y_det.grad)
        param_utils.p_transfer(self.net, self.netB, lambda p1, p2: param_utils.grad_set(p1, p2))

    def step(self, x: IO, t: IO, state: State):
        """Step and zero the gradients.

        Args:
            x (IO): The input.
            t (IO): The target.
            state (State): The learning state.
        """
        self._optim.step()
        self._optim.zero_grad()
        self.netB.zero_grad()


class DFALearner(GradLearner):
    """Learner for implementing direct feedback alignment."""

    def __init__(
        self,
        net: nn.Module,
        netB: nn.Module,
        out_features: int,
        t_features: int,
        optim_factory: OptimFactory,
        activation: nn.Module = None,
        learn_criterion: typing.Union[Criterion, str] = "MSELoss",
    ) -> None:
        """Wrap a network to create a DFALearner.

        It is flexible but somewhat computationally wasteful because it executes forward on netB.

        Args:
            net (nn.Module): The net to use for forward prop. The module must be a single parameterized
                module such as Linear or Conv2d.
            netB (nn.Module): The net to use for backprop. Must have the same architecture as net.
            out_features (int): The number of out features.
            t_features (int): The number of target features.
            optim_factory (OptimFactory): The optimizer.
            activation (nn.Module): The activation. Defaults to None.
            learn_criterion (typing.Union[Criterion, str], optional): The criterion. Defaults to "MSELoss".
        """
        if isinstance(learn_criterion, str):
            learn_criterion = NNLoss(learn_criterion)

        super().__init__(module=net, criterion=learn_criterion)
        self.net = net
        self.netB = netB
        self.activation = activation or Null()
        self.flatten = nn.Flatten()
        self.B = nn.Linear(out_features, t_features, bias=False)
        self._optim = optim_factory(self.net.parameters())

    def forward_nn(self, x: IO, state: State) -> Tensor:
        """Pass the input through the net.

        Args:
            x (IO): The input.
            state (State): The learning state.

        Returns:
            Tensor: The output of the net after activation, and the OutT placeholder.
        """
        y = self.net(x.f)
        y = y.detach()
        state._y_det = y
        y.requires_grad = True
        y.retain_grad()
        y = state._y = self.activation(y)
        state.out_t = OutT()
        return y, state.out_t

    @forward_dep("_y")
    def accumulate(self, x: IO, t: IO, state: State):
        """Update the net parameters.

        Args:
            x (IO): The input.
            t (IO): The target.
            state (State): The learning state.
        """
        if state.out_t.t is None:
            raise RuntimeError("Must set the target of the OutT passed on forward to execute.")

        y2 = self.netB(x.f)

        y_det = state._y_det
        y = state._y
        y = self.B(y.f)
        self.criterion(iou(y), state.out_t.t).backward()
        y2.backward(y_det.grad)

        param_utils.p_transfer(self.net, self.netB, lambda p1, p2: param_utils.grad_set(p1, p2))
        assert x.f.grad is not None

    @forward_dep("_y")
    def step(self, x: IO, t: IO, state: State):
        """Step and zero the gradients.

        Args:
            x (IO): The input.
            t (IO): The target.
            state (State): The learning state.
        """
        self._optim.step()
        self._optim.zero_grad()
        self.netB.zero_grad()

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Update x using the target set on the OutT.

        Args:
            x (IO): The input.
            t (IO): The target.
            state (State): The learning state.

        Returns:
            IO: The updated x.
        """
        if state.out_t.t is None:
            raise RuntimeError("The target for the output has not been set.")

        return super().step_x(x, state.out_t.t, state)
