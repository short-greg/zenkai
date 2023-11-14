# 3rd Party
from zenkai.kaku._assess import Assessment, Criterion, ThLoss

# Local
from ..kaku import IO, State, StepTheta, StepX, LearningMachine


class NullStepTheta(StepTheta):
    """Step that does not update theta"""

    def step(self, x: IO, t: IO, state: State):
        pass


class NullStepX(StepX):
    """Step that does not update theta"""

    def step_x(self, x: IO, t: IO, state: State):
        return x


class NullLearner(NullStepX, NullStepTheta, LearningMachine):
    """'LearningMachine' that does nothing"""

    def __init__(self, criterion: Criterion = None) -> None:
        super().__init__()
        self.criterion = criterion or ThLoss("MSELoss")

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> Assessment:
        return self.criterion.assess(y, t, reduction_override)

    def forward(self, x: IO, state: State, release: bool = True) -> IO:
        return x
