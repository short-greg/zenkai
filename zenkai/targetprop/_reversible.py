# 1st party
import typing

# local
from ._reversible_mods import Reversible, SequenceReversible
from ..kaku import (
    Criterion
)
from ..kaku._state import State
from ..kaku._io2 import (
    IO as IO, iou
)
from ..kaku._lm2 import (
    LearningMachine as LearningMachine,
    StepTheta as StepTheta,
    StepX as StepX,

)


class ReversibleMachine(LearningMachine):
    """Machine that executes a reverse operation to update x"""

    def __init__(
        self,
        reversible: typing.Union[Reversible, typing.List[Reversible]],
        objective: Criterion
    ):
        """Create a machine that will execute a reverse operation

        Args:
            reversible (typing.Union[Reversible, typing.List[Reversible]]): Reversible module to adapt
            loss (ThLoss): The loss
        """
        super().__init__()
        if isinstance(reversible, typing.List):
            reversible = SequenceReversible(*reversible)
        self.reversible = reversible
        self.objective = objective

    def step_x(self, x: IO, t: IO, state: State) -> IO:
        """Update x

        Args:
            x (IO): Input
            
        Returns:
            IO: The updated input
        """
        return iou(self.reversible.reverse(t.f))

    def step(self, x: IO, t: IO, state: State):
        """These layers do not have parameters so the internal mechanics are not updated

        Args:
            x (IO): The input
            t (IO): The output 
        """
        pass

    def forward_nn(self, x: IO, state: State, **kwargs) -> typing.Union[typing.Tuple, typing.Any]:
        return self.reversible(x.f)
