# 1st party
import typing

# 3rd Party
import torch.nn as nn
import torch

# Local
from ..utils import Lambda
from . import IO, LearningMachine, Criterion, ThLoss


class BackTarget(LearningMachine):
    """Use this in general for modules that reshape or
    select elements from the input or when the grad function
    simply reverses the forward operation
    """

    def __init__(
        self,
        module: typing.Union[nn.Module, typing.Callable[[torch.Tensor], torch.Tensor]]=None,
        criterion: Criterion = None,
    ) -> None:
        """Simply "backpropagate" the target. Use if the module is a reshape function

        Args:
            module (typing.Union[nn.Module, typing.Callable[[torch.Tensor], torch.Tensor]]): The module to use
            criterion (Criterion, optional): The criterion for evaluation. Defaults to None.
        """
        super().__init__()
        if module is not None and not isinstance(module, nn.Module):
            module = Lambda(module)
        self.module = module
        self.criterion = criterion or ThLoss("MSELoss")

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        return self.criterion.assess(y, t, reduction_override)

    def forward(self, x: IO, release: bool = True) -> IO:
        x.freshen()
        if self.module is not None:
            y = x._(self).y = self.module(*x.u)
        else:
            y = x._(self).y = x.clone()
        return IO(y).out(release=release)

    def step(self, x: IO, t: IO) -> IO:
        pass

    def step_x(self, x: IO, t: IO) -> IO:

        y = x._(self).y
        y.grad = None
        y.backward(*t.u)
        xs = []
        for x_i in x:
            xs.append(x_i.grad)
            x_i.grad = None
        return IO(*xs, True)
