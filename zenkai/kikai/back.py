import torch.nn as nn
import torch
import typing

from ..kaku import Assessment, LearningMachine, Criterion, ThLoss, State, IO
from ..utils import Lambda


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


def back(f, criterion: Criterion=None) -> BackTarget:
    """Convenicence function to create a back target for cases where
    not much customization is needed. Especially for operations with no parameters
    that are in the middle of the network

    Args:
        f : The Function or NNModule to create a Grad Learner for
        optim (OptimFactory, optional): The optim to use. Defaults to None.
        criterion (Criterion, optional): The criterion. Defaults to None.

    Returns:
        GradLearner: The grad learner to optimize
    """
    if criterion is None:
        criterion = ThLoss('MSELoss', 'mean', weight=0.5)
    if not isinstance(f, nn.Module):
        f = Lambda(f)
    return GradLearner(
        f, criterion, optim, reduction='sum'
    )

