# 1st party
import typing

# 3rd party

# local
from ._lm2 import (
    IO as IO,
    StepX as StepX,
    LMode,
    LearningMachine as LearningMachine
)
from ._state import State
from ..nnz._scikit_mod import ScikitModule


class ScikitLearner(LearningMachine):
    """Machine used to train a Scikit Learn estimator"""

    def __init__(
        self,
        module: ScikitModule,
        step_x: typing.Optional[StepX]=None,
        lmode: LMode=LMode.Standard
    ):
        """Create a machine that wraps the scikit estimator specifying how to update x

        Args:
            module (ScikitEstimator): The
            step_x: The function that does step_x
            loss (Loss): The loss function for the estimator
            preprocessor (nn.Module, optional): Module to preprocess the input sent to the estimator. Defaults to None.
        """
        super().__init__(lmode)
        self._module = module
        self._step_x = step_x

    def step(self, x: IO, t: IO, state: State, **kwargs):
        """Update the estimator

        Args:
            x (IO): Input
            t (IO): Target
        """
        self._module.fit(x.f, t.f, **kwargs)

    def step_x(self, x: IO, t: IO, state: State, **kwargs) -> IO:
        """Update the estimator

        Args:
            x (IO): Input
            t (IO): Target
        Returns:
            IO: the updated x
        """
        if self._step_x is None:
            return x
        return self._step_x.step_x(
            x, state._y, t, state, **kwargs
        )

    def forward_nn(self, x: IO, state: State, **kwargs) -> typing.Union[typing.Tuple, typing.Any]:
        """Pass through the scikit estimator

        Args:
            x (IO): The input
            state (State): The learning state

        Returns:
            typing.Union[typing.Tuple, typing.Any]: The output
        """
        return self._module(x[0]).type_as(x[0])

    @property
    def fitted(self) -> bool:
        """
        Returns:
            bool: Whether or not the estimator has been fitted
        """
        return self._module.fitted
