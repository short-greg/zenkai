# 1st party
import typing

# 3rd Party
import torch.nn as nn
import torch

from zenkai.kaku.io import IO
from zenkai.kaku.state import State

from .. import kaku
from ..utils import module_factory, Lambda

# Local
from ..kaku import (
    IO,
    BatchIdxStepTheta,
    BatchIdxStepX,
    Idx,
    LearningMachine,
    State,
    Criterion,
    idx_io,
    Assessment,
    optimf,
    Criterion,
    acc_dep,
    step_dep,
    ThLoss
)
from ..utils import get_model_grads, set_model_grads, Null

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