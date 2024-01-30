# 1st party
import typing

# 3rd party
import torch.nn as nn
import torch

# local
from ..kaku import (
    IO,
    State,
    LearningMachine,
    Assessment,
    OptimFactory,
    Criterion,
    ThLoss,
    Builder,
    UNDEFINED,
    Var,
    Factory,
)
from ._grad import GradUpdater
from ..mod import Null


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


class FALearner(LearningMachine):
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
        super().__init__()
        self.net = net
        self.netB = netB
        self.activation = activation or Null()
        self.flatten = nn.Flatten()
        self._optim = optim_factory(self.net.parameters())

        self._grad_updater = GradUpdater(self.netB, self._optim)
        if isinstance(criterion, str):
            self.criterion = ThLoss(criterion)
        else:
            self.criterion = criterion

    def forward(self, x: IO, state: State, release: bool = True) -> IO:

        x.freshen()
        y = self.net(x.f)
        y = y.detach()
        x._.y_det = y
        # state[self, x, "y_det"] = y
        y.requires_grad = True
        y.retain_grad()
        y = x._.y = self.activation(y)
        # y = state[self, x, "y"] = self.activation(y)
        return IO(y).out(release)

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self.criterion.assess(y, t, reduction_override)

    def accumulate(self, x: IO, t: IO, state: State):
        """Update the

        Args:
            x (IO): the input
            t (IO[y, y_prime]): the target
            state (State): the learning state

        Returns:
            IO: the updated target
        """
        #  my_state = state.mine(self, x)
        self.net.zero_grad()
        self.netB.zero_grad()

        if "y" not in x._:
            self(x, state=state)

        # y = state[self, x, "y"]
        y = x._.y
        y2 = self.netB(x.f)

        self.criterion(IO(y), t).backward()
        y_det = x._.y_det
        # y_det = state[self, x, "y_det"]
        y2.backward(y_det.grad)

        self._grad_updater.accumulate(x, state)

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Backpropagates the error resulting from the randomly generated matrix

        Args:
            x (IO): the input
            t (IO[y, y_prime]): the target
            state (State): the learning state

        Returns:
            IO: the updated target
        """
        x_prime, _ = self._grad_updater.update_x(x, state)
        return x_prime

    def step(self, x: IO, t: typing.Union[IO, None], state: State) -> bool:
        """Advance the optimizer

        Returns:
            bool: False if unable to advance (already advanced or not stepped yet)
        """
        return self._grad_updater.update(x, state, self.net)

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


class DFALearner(LearningMachine):
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
        super().__init__()
        self.net = net
        self.netB = netB
        self.activation = activation or Null()
        self.flatten = nn.Flatten()
        self.B = nn.Linear(out_features, t_features, bias=False)
        self._optim = optim_factory(self.net.parameters())
        if isinstance(criterion, str):
            self.criterion = ThLoss(criterion)
        else:
            self.criterion = criterion
        self._grad_updater = GradUpdater(self.netB, self._optim)

    def forward(self, x: IO, state: State, release: bool = True) -> IO:

        x.freshen()
        y = self.net(x.f)
        y = y.detach()
        x._.y_det = y
        # state[self, x, "y_det"] = y
        y.requires_grad = True
        y.retain_grad()
        y = x._.y = self.activation(y)
        # y = state[self, x, "y"] = self.activation(y)
        return IO(y).out(release)

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self.criterion.assess(y, t, reduction_override)

    def accumulate(self, x: IO, t: IO, state: State):
        """Update the net parameters

        Args:
            x (IO[x]): the input
            t (IO[y]): the target
            state (State): the learning state

        Returns:
            IO: the updated target
        """
        # my_state = state.mine(self, x)
        self.net.zero_grad()
        self.netB.zero_grad()
        self.B.zero_grad()
        if "y" not in x._:
            self(x, state=state)

        y2 = self.netB(x.f)

        # y_det = state[self, x, "y_det"]
        y_det = x._.y_det
        y = x._.y
        # y = state[self, x, "y"]
        y = self.B(y)
        self.criterion(IO(y), t).backward()
        y2.backward(y_det.grad)

        assert x.f.grad is not None
        self._grad_updater.accumulate(x, state)

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Backpropagates the error resulting from the randomly generated matrix

        Args:
            x (IO): the input
            t (IO[y, y_prime]): the target
            state (State): the learning state

        Returns:
            IO: the updated target
        """
        x_prime, _ = self._grad_updater.update_x(x, state)
        return x_prime

    def step(self, x: IO, t: typing.Union[IO, None], state: State) -> bool:
        """Advance the optimizer

        Returns:
            bool: False if unable to advance (already advanced or not stepped yet)
        """

        return self._grad_updater.update(x, state, self.net)


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
