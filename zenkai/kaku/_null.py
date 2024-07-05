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

    def forward_nn(self, x: IO, state: State, **kwargs) -> typing.Union[typing.Tuple, typing.Any]:
        """Does nothing as it is a "null learner"

        Args:
            x (IO): The input
            state (State): The learning state

        Returns:
            typing.Union[typing.Tuple, typing.Any]: The input
        """
        return x.f
