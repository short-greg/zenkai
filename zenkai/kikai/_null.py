# 3rd Party
import torch

# Local
from ..kaku import IO, StepTheta, StepX, LearningMachine
from ..kaku._assess import Criterion, ThLoss


class NullStepTheta(StepTheta):
    """Step that does not update theta"""

    def step(self, x: IO, t: IO):
        pass


class NullStepX(StepX):
    """Step that does not update theta"""

    def step_x(self, x: IO, t: IO):
        return x


class NullLearner(NullStepX, NullStepTheta, LearningMachine):
    """'LearningMachine' that does nothing"""

    def __init__(self, criterion: Criterion = None) -> None:
        super().__init__()
        self.criterion = criterion or ThLoss("MSELoss")

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        return self.criterion.assess(y, t, reduction_override)

    def forward(self, x: IO, release: bool = True) -> IO:
        return x
