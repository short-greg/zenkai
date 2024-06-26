# 1st Party
import typing

# 3rd Party
import torch

# Local
from ._lm2 import IO as IO, StepTheta as StepTheta, StepX as StepX, LearningMachine as LearningMachine
from ._assess import Criterion, NNLoss
from ._state import State


class NullStepTheta(StepTheta):
    """Step that does not update theta"""

    def step(self, x: IO, t: IO, state: State, **kwargs):
        pass


class NullStepX(StepX):
    """Step that does not update theta"""

    def step_x(self, x: IO, t: IO, state: State, **kwargs):
        return x


class NullLearner(NullStepX, NullStepTheta, LearningMachine):
    """'LearningMachine' that does nothing"""

    def __init__(self, criterion: Criterion = None) -> None:
        super().__init__()
        self.criterion = criterion or NNLoss("MSELoss")

    # def assess_y(self, y: IO, t: IO, state: State, reduction_override: str = None) -> torch.Tensor:
    #     """Assess the output

    #     Args:
    #         y (IO): The output
    #         t (IO): The target
    #         state (State): The learning state
    #         reduction_override (str, optional): Defaults to None.

    #     Returns:
    #         torch.Tensor: _description_
    #     """
    #     return self.criterion.assess(y, t, reduction_override)

    def forward_nn(self, x: IO, state: State, **kwargs) -> typing.Union[typing.Tuple, typing.Any]:
        """Does nothing as it is a "null learner"

        Args:
            x (IO): The input
            state (State): The learning state

        Returns:
            typing.Union[typing.Tuple, typing.Any]: The input
        """
        return x.f
