# 1st party
import typing

# local
from ..kaku import IO, Assessment, IO, LearningMachine, State, Criterion
from ..utils import Reversible, SequenceReversible
import torch.nn as nn
import torch
import typing

from ..kaku import Assessment, LearningMachine, Criterion, ThLoss, State, IO
from ..utils import Lambda



class ReversibleMachine(LearningMachine):
    """Machine that executes a reverse operation to update x"""

    def __init__(
        self,
        reversible: typing.Union[Reversible, typing.List[Reversible]],
        objective: Criterion
    ):
        """initializer

        Args:
            reversible (typing.Union[Reversible, typing.List[Reversible]]): Reversible module to adapt
            loss (ThLoss): The loss
        """
        super().__init__()
        if isinstance(reversible, typing.List):
            reversible = SequenceReversible(*reversible)
        self.reversible = reversible
        self.objective = objective

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self.objective.assess(y, t, reduction_override)

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Update x

        Args:
            x (IO): Input
            state (State): The learning state

        Returns:
            IO: The updated input
        """
        return IO(self.reversible.reverse(t.f), detach=True)

    def step(self, x: IO, t: IO, state: State):
        """These layers do not have parameters so the internal mechanics are not updated

        Args:
            x (IO): The input
            t (IO): The output 
            state (State): The learning state
        """
        pass

    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        return IO(self.reversible(x.f)).out(release)


class BackTarget(LearningMachine):
    """Use this in general for modules that reshape or 
    select elements from the input or when the grad function
    simply reverses the forward operation
    """

    def __init__(self, module: typing.Union[nn.Module, typing.Callable[[torch.Tensor], torch.Tensor]], criterion: Criterion=None) -> None:
        super().__init__()
        self.module = module if isinstance(module, nn.Module) else Lambda(module)
        self.criterion = criterion or ThLoss('mse')

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self.criterion.assess(y, t, reduction_override)

    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        x.freshen()
        y = state[(self, x), 'y'] = self.module(*x.u)
        return IO(y).out(release=release)

    def step(self, x: IO, t: IO, state: State) -> IO:
        pass

    def step_x(self, x: IO, t: IO, state: State) -> IO:

        y = state[(self, x), 'y']
        y.grad = None
        y.backward(*t.u)
        xs = []
        for x_i in x:
            xs.append(x_i.grad)
            x_i.grad = None
        return IO(*xs, True)


def reverse(f, criterion: Criterion=None) -> typing.Union[ReversibleMachine, BackTarget]:
    """Convenicence function to create a reverse for cases where
    not much customization is needed. Especially for operations that do not
    have parameters and they can either be reversed through the backward operation or
    through a 'reverse' method. If it is a Reversible then a ReversibleMachine will be created. Otherwise,
    a BackTarget will be created

    Args:
        f : The Function or NNModule to create a BackTarget or ReversibleMachine for
        criterion (Criterion, optional): The criterion. Defaults to None.

    Returns:
        typing.Union[ReversibleMachine, BackTarget]: The Reversible machine to optimize
    """
    if criterion is None:
        criterion = ThLoss('MSELoss', 'mean', weight=0.5)
    if isinstance(f, Reversible):
        return ReversibleMachine(
            f, criterion
        )

    if not isinstance(f, nn.Module):
        f = Lambda(f)
    return BackTarget(
        f, criterion, reduction='sum'
    )
