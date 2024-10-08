# 1st party
import typing
from dataclasses import dataclass

# 3rd party
from torch._tensor import Tensor
import torch.nn as nn
import torch

# local
from ..kaku import (
    OptimFactory,
    Criterion,
    NNLoss,
)
from ..utils._build import (
    Builder,
    UNDEFINED,
    Var,
    Factory,
)
from ..utils import _params as param_utils
from ..targetprop import Null
from ..kaku._grad import GradLearner
from ..kaku._lm2 import IO as IO, Idx as Idx, forward_dep
from ..kaku._io2 import IO as IO, iou
from ..kaku import Idx
from ..kaku._state import State
from ..kaku._lm2 import LearningMachine as LearningMachine


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
            learn_criterion=learn_criterion
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
            batch_idx (Idx, optional): The index to use. Defaults to None.

        Returns:
            torch.Tensor: The 
        """

        y = self.net(x.f)
        y = y.detach()
        state._y_det = y
        y.requires_grad = True
        y.retain_grad()
        return self.activation(y)

    @forward_dep('_y')
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
        
        self._learn_criterion.assess(y, t).backward()
        y_det = state._y_det
        y2.backward(y_det.grad)
        param_utils.set_model_grads(
            self.net, param_utils.get_model_grads(self.netB)
        )
    
    @forward_dep('_y')
    def step(self, x: IO, t: IO, state: State):
        self._optim.step()
        self._optim.zero_grad()
        self.netB.zero_grad()

    @classmethod
    def builder(
        cls,
        net=UNDEFINED,
        netB=UNDEFINED,
        optim_factory=UNDEFINED,
        activation="ReLU",
        learn_criterion="MSELoss",
    ) -> Builder["FALearner"]:

        """ """
        kwargs = Builder.kwargs(
            net=net,
            netB=netB,
            activation=activation,
            learn_criterion=learn_criterion,
            optim_factory=optim_factory,
        )

        return Builder[FALearner](
            FALearner,
            ["net", "netB", "optim_factory", "activation", "learn_criterion"],
            **kwargs
        )

@dataclass
class OutT:
    """Use to store the value to set T to be
    """

    t: IO = None


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
            learn_criterion=learn_criterion
        )
        self.net = net
        self.netB = netB
        self.activation = activation or Null()
        self.flatten = nn.Flatten()
        self.B = nn.Linear(out_features, t_features, bias=False)
        self._optim = optim_factory(self.net.parameters())

    def forward_nn(self, x: IO, state: State,  out_t: OutT=None) -> Tensor:

        y = self.net(x.f)
        y = y.detach()
        state._y_det = y
        y.requires_grad = True
        y.retain_grad()
        return self.activation(y)

    @forward_dep('_y')
    def accumulate(self, x: IO, t: IO, state: State, out_t: OutT=None):
        """Update the net parameters

        Args:
            x (IO[x]): the input
            t (IO[y]): the target

        Returns:
           
            IO: the updated target
        """
        if out_t.t is None:
            # TODO: Add warning
            return

        y2 = self.netB(x.f)

        y_det = state._y_det
        y = state._y
        y = self.B(y.f)
        self._learn_criterion(iou(y), out_t.t).backward()
        y2.backward(y_det.grad)

        param_utils.set_model_grads(self.net, param_utils.get_model_grads(self.netB))
        assert x.f.grad is not None
    
    @forward_dep('_y')
    def step(self, x: IO, t: IO, state: State, out_t: OutT=None):
        self._optim.step()
        self._optim.zero_grad()
        self.netB.zero_grad()

    def step_x(self, x: IO, t: IO, state: State, out_t: OutT) -> IO:
        
        return super().step_x(
            x, out_t.t, state
        )


class LinearFABuilder(Builder[FALearner]):
    """Learner for implementing feedback alignment."""
    def __init__(
        self,
        in_features: int = UNDEFINED,
        out_features: int = UNDEFINED,
        optim_factory: OptimFactory = UNDEFINED,
        activation: nn.Module = UNDEFINED,
        learn_criterion: Criterion = UNDEFINED,
    ):
        """Create a builder for LinearFABuilder

        Args:
            in_features (int, optional): The linear in features. Defaults to UNDEFINED.
            out_features (int, optional): The number out features. Defaults to UNDEFINED.
            optim_factory (OptimFactory, optional): The optimizer to use. Defaults to UNDEFINED.
            activation (nn.Module, optional): The activation for the feedback alignment. Defaults to UNDEFINED.
            learn_criterion (Criterion, optional): The learn_criterion to use for optimizing. Defaults to UNDEFINED.
        """
        
        super().__init__(
            FALearner,
            ["in_features", "out_features", "optim_factory", "activation", "learn_criterion"],
            net=Factory(
                nn.Linear,
                Var.init("in_features", in_features),
                Var.init("out_features", out_features),
            ),
            netB=Factory(
                nn.Linear,
                Var("in_features", in_features),
                Var("out_features", out_features),
            ),
            optim_factory=Var.init("optim_factory", optim_factory),
            activation=Factory(Var.init("activation", activation)),
            learn_criterion=Var.init("learn_criterion", learn_criterion),
        )
        self.in_features = self.Updater(self, "in_features")
        self.out_features = self.Updater(self, "out_features")
        self.optim_factory = self.Updater(self, "optim_factory")
        self.learn_criterion = self.Updater(self, "learn_criterion")
        self.activation = self.Updater(self, "activation")


class LinearDFABuilder(Builder[DFALearner]):
    """Builds a linear DFA
    """
    
    def __init__(
        self,
        in_features: int = UNDEFINED,
        out_features: int = UNDEFINED,
        t_features: int = UNDEFINED,
        optim_factory: OptimFactory = UNDEFINED,
        activation: nn.Module = UNDEFINED,
        learn_criterion: Criterion = UNDEFINED,
    ):
        """Create to build a LinearDFA

        Args:
            in_features (int, optional): The in features to the DFA. Defaults to UNDEFINED.
            out_features (int, optional): The out features to the DFA. Defaults to UNDEFINED.
            t_features (int, optional): The number of features for the target layer. Defaults to UNDEFINED.
            optim_factory (OptimFactory, optional): The optimizer to use. Defaults to UNDEFINED.
            activation (nn.Module, optional): The activation to use if any. Defaults to UNDEFINED.
            learn_criterion (Criterion, optional): The criterion to use if any. Defaults to UNDEFINED.
        """

        super().__init__(
            DFALearner,
            ["in_features", "out_features", "optim_factory", "activation", "learn_criterion"],
            net=Factory(
                nn.Linear,
                Var.init("in_features", in_features),
                Var.init("out_features", out_features),
            ),
            netB=Factory(
                nn.Linear,
                Var.init("in_features"),
                Var.init("out_features", out_features),
            ),
            out_features=Var('out_features'),
            optim_factory=Var.init("optim_factory", optim_factory),
            activation=Factory(Var.init("activation", activation)),
            t_features=Var.init("t_features", t_features),
            learn_criterion=Var.init("learn_criterion", learn_criterion),
        )
        self.in_features = self.Updater[LinearDFABuilder, int](self, "in_features")
        self.out_features = self.Updater[LinearDFABuilder, int](self, "out_features")
        self.t_features = self.Updater[LinearDFABuilder, int](self, "t_features")
        self.optim_factory = self.Updater[LinearDFABuilder, OptimFactory](
            self, "optim_factory"
        )
        self.learn_criterion = self.Updater[LinearDFABuilder, Criterion](self, "learn_criterion")
        self.activation = self.Updater[LinearDFABuilder, typing.Type[nn.Module]](
            self, "activation"
        )
