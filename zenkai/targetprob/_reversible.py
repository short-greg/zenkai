# 1st party
import typing

# 3rd party
import torch

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
        """initializer

        Args:
            reversible (typing.Union[Reversible, typing.List[Reversible]]): Reversible module to adapt
            loss (ThLoss): The loss
        """
        super().__init__()
        if isinstance(reversible, typing.List):
            reversible = SequenceReversible(*reversible)
        self.reversible = reversible
        self.objective = objective

    def assess_y(self, y: IO, t: IO, reduction_override: str = None) -> torch.Tensor:
        return self.objective.assess(y, t, reduction_override)

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
    # def forward(self, x: IO, release: bool = True) -> IO:
    #     return iou(self.reversible(x.f))


# def reverse(f, criterion: Criterion = None) -> typing.Union[ReversibleMachine, BackTarget]:
#     """Convenicence function to create a reverse for cases where
#     not much customization is needed. Especially for operations that do not
#     have parameters and they can either be reversed through the backward operation or
#     through a 'reverse' method. If it is a Reversible then a ReversibleMachine will be created. Otherwise,
#     a BackTarget will be created

#     Args:
#         f : The Function or NNModule to create a BackTarget or ReversibleMachine for
#         criterion (Criterion, optional): The criterion. Defaults to None.

#     Returns:
#         typing.Union[ReversibleMachine, BackTarget]: The Reversible machine to optimize
#     """
#     if criterion is None:
#         criterion = ThLoss('MSELoss', 'mean', weight=0.5)
#     if isinstance(f, Reversible):
#         return ReversibleMachine(
#             f, criterion
#         )

#     if not isinstance(f, nn.Module):
#         f = Lambda(f)
#     return BackTarget(
#         f, criterion
#     )
