# 1st party
import typing

# 3rd party
import torch.nn as nn

from ..kaku import Idx

# local
from ..kaku import (
    IO,
    OptimFactory,
    Criterion,
    ThLoss,
    Builder,
    UNDEFINED,
    Var,
    Factory,
    forward_dep
)
from .. import utils
from ..targetprob import Null
from ..kaku._grad import GradLearner


def fa_target(y: IO, y_prime: IO, detach: bool = True) -> IO:
    """create the target for feedback alignment

    Args:
        y (IO): The original output of the layer
        y_prime (IO): The updated target
        detach (bool, optional): whether to detach. Defaults to True.

    Returns:
        IO: the resulting target
    """

    return IO(y.f, y_prime.f, detach=detach)


class FALearner(GradLearner):
    """Learner for implementing feedback alignment"""

    def __init__(
        self,
        net: nn.Module,
        netB: nn.Module,
        optim_factory: OptimFactory,
        activation: nn.Module = None,
        criterion: typing.Union[Criterion, str] = "MSELoss",
    ) -> None:
        """Wraps a module to create an FALearner.
        It flexible but somewhat computationally wasteful because it executes forward on netB

        Args:
            net (nn.Module): the net to use for forward prop
            netB (nn.Module): the net to use for backprop
            optim_factory (OptimFactory): The opimtizer
            activation (nn.Module): The activation
            criterion (typing.Union[Criterion, str], optional): The criterion. Defaults to 'mse'.
        """
        if isinstance(criterion, str):
            criterion = ThLoss(criterion)

        super().__init__(
            module=net,
            optim=optim_factory.comp(), 
            criterion=criterion
        )
        self.net = net
        self.netB = netB
        self.activation = activation or Null()
        self.flatten = nn.Flatten()
        
    def forward_nn(self, x: IO) -> IO:

        y = self.net(x.f)
        y = y.detach()
        x._(self).y_det = y
        y.requires_grad = True
        y.retain_grad()
        return IO(self.activation(y))

    @forward_dep('y')
    def accumulate(self, x: IO, t: IO):
        """Update the

        Args:
            x (IO): the input
            t (IO[y, y_prime]): the target

        Returns:
            IO: the updated target
        """
        self.optim.prep_x(x)
        y = x._(self).y
        y2 = self.netB(x.f)
        
        self.criterion.assess(y, t).backward()
        y_det = x._(self).y_det
        y2.backward(y_det.grad)
        utils.set_model_grads(self.net, utils.get_model_grads(self.netB))
    
    @forward_dep('y')
    def step(self, x: IO, t: IO, batch_idx: Idx = None):
        super().step(x, t, batch_idx)
        self.netB.zero_grad()

    @classmethod
    def builder(
        cls,
        net=UNDEFINED,
        netB=UNDEFINED,
        optim_factory=UNDEFINED,
        activation="ReLU",
        criterion="MSELoss",
    ) -> Builder["FALearner"]:

        """ """
        kwargs = Builder.kwargs(
            net=net,
            netB=netB,
            activation=activation,
            criterion=criterion,
            optim_factory=optim_factory,
        )

        return Builder[FALearner](
            FALearner,
            ["net", "netB", "optim_factory", "activation", "criterion"],
            **kwargs
        )


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
        criterion: typing.Union[Criterion, str] = "MSELoss",
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
        if isinstance(criterion, str):
            criterion = ThLoss(criterion)
        
        super().__init__(
            module=net,
            optim=optim_factory.comp(),
            criterion=criterion
        )
        self.net = net
        self.netB = netB
        self.activation = activation or Null()
        self.flatten = nn.Flatten()
        self.B = nn.Linear(out_features, t_features, bias=False)

    def forward_nn(self, x: IO) -> IO:

        y = self.net(x.f)
        y = y.detach()
        x._(self).y_det = y
        y.requires_grad = True
        y.retain_grad()
        return IO(self.activation(y))

    @forward_dep('y')
    def accumulate(self, x: IO, t: IO):
        """Update the net parameters

        Args:
            x (IO[x]): the input
            t (IO[y]): the target

        Returns:
           
            IO: the updated target
        """
        self.optim.prep_x(x)
        y2 = self.netB(x.f)

        y_det = x._(self).y_det
        y = x._(self).y
        y = self.B(y.f)
        self.criterion(IO(y), t).backward()
        y2.backward(y_det.grad)

        utils.set_model_grads(self.net, utils.get_model_grads(self.netB))
        assert x.f.grad is not None
    
    @forward_dep('y')
    def step(self, x: IO, t: IO, batch_idx: Idx = None):
        
        super().step(x, t, batch_idx)
        self.netB.zero_grad()


class LinearFABuilder(Builder[FALearner]):
    def __init__(
        self,
        in_features: int = UNDEFINED,
        out_features: int = UNDEFINED,
        optim_factory: OptimFactory = UNDEFINED,
        activation: nn.Module = UNDEFINED,
        criterion: Criterion = UNDEFINED,
    ):

        super().__init__(
            FALearner,
            ["in_features", "out_features", "optim_factory", "activation", "criterion"],
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
            criterion=Var.init("criterion", criterion),
        )
        self.in_features = self.Updater(self, "in_features")
        self.out_features = self.Updater(self, "out_features")
        self.optim_factory = self.Updater(self, "optim_factory")
        self.criterion = self.Updater(self, "criterion")
        self.activation = self.Updater(self, "activation")


class LinearDFABuilder(Builder[DFALearner]):
    def __init__(
        self,
        in_features: int = UNDEFINED,
        out_features: int = UNDEFINED,
        t_features: int = UNDEFINED,
        optim_factory: OptimFactory = UNDEFINED,
        activation: nn.Module = UNDEFINED,
        criterion: Criterion = UNDEFINED,
    ):

        super().__init__(
            DFALearner,
            ["in_features", "out_features", "optim_factory", "activation", "criterion"],
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
            criterion=Var.init("criterion", criterion),
        )
        self.in_features = self.Updater[LinearDFABuilder, int](self, "in_features")
        self.out_features = self.Updater[LinearDFABuilder, int](self, "out_features")
        self.t_features = self.Updater[LinearDFABuilder, int](self, "t_features")
        self.optim_factory = self.Updater[LinearDFABuilder, OptimFactory](
            self, "optim_factory"
        )
        self.criterion = self.Updater[LinearDFABuilder, Criterion](self, "criterion")
        self.activation = self.Updater[LinearDFABuilder, typing.Type[nn.Module]](
            self, "activation"
        )
