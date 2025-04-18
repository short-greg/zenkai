# 1st party
import typing
from dataclasses import dataclass

# 3rd party
from torch._tensor import Tensor
import torch.nn as nn
import torch

# local
from . import (
    Criterion, NNLoss,
)
from ..optimz import OptimFactory
from ..utils import _params as param_utils
from ..nnz import Null
from ._grad import GradLearner
from . import GradLearner, IO, iou, forward_dep, State
from ._lm2 import LearningMachine as LearningMachine, OutT


def fa_target(y: IO, y_prime: IO, detach: bool = True) -> IO:
    """create the target for feedback alignment

    Args:
        y (IO): The original output of the layer
        y_prime (IO): The updated target
        detach (bool, optional): whether to detach. Defaults to True.

    Returns:
        IO: the resulting target
    """
    return iou(y.f, y_prime.f, detach=detach)


class FALearner(GradLearner):
    """Learner for implementing feedback alignment"""

    def __init__(
        self,
        net: nn.Module,
        netB: nn.Module,
        activation: nn.Module = None,
        optim_factory: OptimFactory = None,
        learn_criterion: typing.Union[Criterion, str] = "MSELoss",
    ) -> None:
        """Wraps a module to create an FALearner.
        It flexible but somewhat computationally wasteful because it executes forward on netB

        Args:
            net (nn.Module): the net to use for forward prop
            netB (nn.Module): the net to use for backprop
            optim_factory (OptimFactory): The opimtizer
            activation (nn.Module): The activation
            learn_criterion (typing.Union[Criterion, str], optional): The criterion. Defaults to 'mse'.
        """
        if isinstance(learn_criterion, str):
            learn_criterion = NNLoss(learn_criterion)

        super().__init__(
            module=net,
            criterion=learn_criterion
        )
        self.net = net
        self.netB = netB
        self.activation = activation or Null()
        self.flatten = nn.Flatten()
        self._optim = optim_factory(self.net.parameters())
        
    def forward_nn(self, x: IO, state: State) -> torch.Tensor:
        """Pass the input through the net

        Args:
            x (IO): The input
            state (State): The learning state
            Defaults to None.

        Returns:
            torch.Tensor: The 
        """
        y = self.net(x.f)
        y = y.detach()
        state._y_det = y
        y.requires_grad = True
        y.retain_grad()
        return self.activation(y)

    def accumulate(self, x: IO, t: IO, state: State):
        """Accumulate the gradients

        Args:
            x (IO): the input
            t (IO[y, y_prime]): the target

        Returns:
            IO: the updated target
        """
        y = state._y
        y2 = self.netB(x.f)
        
        self.criterion.assess(y, t).backward()
        y_det = state._y_det
        y2.backward(y_det.grad)
        param_utils.transfer_p(
            self.net, self.netB,
            lambda p1, p2: param_utils.set_grad(p1, p2)
        )
        # param_utils.set_model_grads(
        #     self.net, param_utils.get_model_grads(self.netB)
        # )
    
    def step(self, x: IO, t: IO, state: State):
        self._optim.step()
        self._optim.zero_grad()
        self.netB.zero_grad()


class DFALearner(GradLearner):
    """Learner for implementing feedback alignment."""

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
        """Wraps a network to create a DFALearner.
        It flexible but somewhat computationally wasteful because it executes forward on netB

        Args:
            net (nn.Module): the net to use for forward prop. The module must be a
                single paramterized module such as Linear, or Conv2d
            netB (nn.Module): the net to use for backprop. Must be have the same architecture as net
            out_features (int): the number of out features
            t_features (int): the number of target features
            optim_factory (OptimFactory): The opimtizer
            activation (nn.Module): The activation
            criterion (typing.Union[Criterion, str], optional): The criterion. 
                Defaults to 'mse'.
        """
        if isinstance(learn_criterion, str):
            learn_criterion = NNLoss(learn_criterion)
        
        super().__init__(
            module=net,
            criterion=learn_criterion
        )
        self.net = net
        self.netB = netB
        self.activation = activation or Null()
        self.flatten = nn.Flatten()
        self.B = nn.Linear(out_features, t_features, bias=False)
        self._optim = optim_factory(self.net.parameters())

    def forward_nn(self, x: IO, state: State) -> Tensor:

        y = self.net(x.f)
        y = y.detach()
        state._y_det = y
        y.requires_grad = True
        y.retain_grad()
        y = state._y = self.activation(y)
        state.out_t = OutT()
        return y, state.out_t

    @forward_dep('_y')
    def accumulate(self, x: IO, t: IO, state: State):
        """Update the net parameters

        Args:
            x (IO[x]): the input
            t (IO[y]): the target

        Returns:
           
            IO: the updated target
        """
        if state.out_t.t is None:
            # TODO: Add warning
            raise RuntimeError(
                'Must set the target of the OutT passed '
                'on forward to execute.'
            )

        y2 = self.netB(x.f)

        y_det = state._y_det
        y = state._y
        y = self.B(y.f)
        self.criterion(iou(y), state.out_t.t).backward()
        y2.backward(y_det.grad)

        # param_utils.set_model_grads(self.net, param_utils.get_model_grads(self.netB))
        param_utils.transfer_p(
            self.net, self.netB,
            lambda p1, p2: param_utils.set_grad(p1, p2)
        )
        assert x.f.grad is not None
    
    @forward_dep('_y')
    def step(self, x: IO, t: IO, state: State):
        self._optim.step()
        self._optim.zero_grad()
        self.netB.zero_grad()

    def step_x(self, x: IO, t: IO, state: State) -> IO:

        if state.out_t.t is None:
            raise RuntimeError(
                'The target for the output has not been set.'
            )
        
        return super().step_x(
            x, state.out_t.t, state
        )
