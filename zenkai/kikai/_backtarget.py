# 1st party
import typing

# 3rd Party
import torch.nn as nn
import torch

# Local
from ..mod import Lambda
from ..kaku import IO, LearningMachine, State, Criterion, Assessment, ThLoss


class BackTarget(LearningMachine):
    """Use this in general for modules that reshape or
    select elements from the input or when the grad function
    simply reverses the forward operation
    """

    def __init__(
        self,
        module: typing.Union[nn.Module, typing.Callable[[torch.Tensor], torch.Tensor]],
        criterion: Criterion = None,
    ) -> None:
        super().__init__()
        self.module = module if isinstance(module, nn.Module) else Lambda(module)
        self.criterion = criterion or ThLoss("MSELoss")

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self.criterion.assess(y, t, reduction_override)

    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        x.freshen()
        y = state[self, x, "y"] = self.module(*x.u)
        return IO(y).out(release=release)

    def step(self, x: IO, t: IO, state: State) -> IO:
        pass

    def step_x(self, x: IO, t: IO, state: State) -> IO:

        y = state[self, x, "y"]
        y.grad = None
        y.backward(*t.u)
        xs = []
        for x_i in x:
            xs.append(x_i.grad)
            x_i.grad = None
        return IO(*xs, True)
