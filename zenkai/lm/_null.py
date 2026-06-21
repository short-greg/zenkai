# 1st Party
import typing

# Local
from zenkai._core import IO, State

from ._lm import LearningMachine, StepTheta, StepX


class NullStepTheta(StepTheta):
    """Step that does not update theta"""

    def step(self, x: IO, y: IO, t: IO, state: State, **kwargs):
        pass


class NullStepX(StepX):
    """Step that does not update theta"""

    def step_x(self, x: IO, y: IO, t: IO, state: State, **kwargs):
        return x


class NullLearner(LearningMachine):
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

    def step(self, x: IO, t: IO, state: State, **kwargs):
        pass

    def step_x(self, x: IO, t: IO, state: State, **kwargs):
        return x
