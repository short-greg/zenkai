# 1st Party
import typing

# 3rd Party
import torch

# Local
from ._lm2 import IO2 as IO, StepTheta2 as StepTheta, StepX2 as StepX, LM as LearningMachine
from ._assess import Criterion, ThLoss
from ._state import Meta


class NullStepTheta(StepTheta):
    """Step that does not update theta"""

    def step(self, x: IO, t: IO, state: Meta, **kwargs):
        pass


class NullStepX(StepX):
    """Step that does not update theta"""

    def step_x(self, x: IO, t: IO, state: Meta, **kwargs):
        return x


class NullLearner(NullStepX, NullStepTheta, LearningMachine):
    """'LearningMachine' that does nothing"""

    def __init__(self, criterion: Criterion = None) -> None:
        super().__init__()
        self.criterion = criterion or ThLoss("MSELoss")

    def assess_y(self, y: IO, t: IO, state: Meta, reduction_override: str = None) -> torch.Tensor:
        return self.criterion.assess(y, t, reduction_override)

    def forward_nn(self, x: IO, state: Meta, **kwargs) -> typing.Union[typing.Tuple, typing.Any]:
        return x
