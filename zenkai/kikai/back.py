import torch.nn as nn
from zenkai.kaku.io import IO
from zenkai.kaku.state import State

from ..kaku import Assessment, LearningMachine, Criterion, ThLoss


class BackTarget(LearningMachine):
    """Use this in general for modules that reshape or 
    select elements from the input or when the grad function
    simply reverses the forward operation
    """

    def __init__(self, module: nn.Module, criterion: Criterion=None) -> None:
        super().__init__()
        self.module = module
        self.criterion = criterion or ThLoss('mse')

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self.criterion.assess(y, t, reduction_override)

    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        x.freshen()
        y = state[(self, x), 'y'] = self.module(x.f)
        return IO(y).out(release=release)

    def step(self, x: IO, t: IO, state: State) -> IO:
        pass

    def step_x(self, x: IO, t: IO, state: State) -> IO:

        y = state[(self, x), 'y']
        y.grad = None
        y.backward(t.f)
        return IO(y.grad, True)
