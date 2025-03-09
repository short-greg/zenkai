# 1st party
from abc import abstractmethod

# local
from ._state import State
from ._io2 import (
    IO as IO
)
from ._lm2 import (
    LearningMachine as LearningMachine,
    StepTheta as StepTheta,
    StepX as StepX,
)


class ReverseLearner(LearningMachine):
    """
    ReverseLearner is a learner that is designed to reverse an operation.
    The first value in the IO `x` is assumed to be the input to the machine to reverse,
    and the second value is assumed to be its output. Therefore, `x` is an IO whose 
    first two elements are IOs.
    """
    @abstractmethod
    def forward_nn(self, x: IO, state: State, **kwargs) -> IO:
        pass

    @abstractmethod
    def step_x(self, x: IO, t: IO, state: State, **kwargs) -> IO:
        pass

    @abstractmethod
    def accumulate(self, x, t, state, **kwargs):
        pass

    @abstractmethod
    def step(self, x, t, state, **kwargs):
        pass


# class ReversibleMachine(LearningMachine):
#     """Machine that executes a reverse operation to update x"""

#     def __init__(
#         self,
#         reversible: typing.Union[Reversible, typing.List[Reversible]],
#     ):
#         """Create a machine that will execute a reverse operation

#         Args:
#             reversible (typing.Union[Reversible, typing.List[Reversible]]): Reversible module to adapt
#             loss (ThLoss): The loss
#         """
#         super().__init__()
#         if isinstance(reversible, typing.List):
#             reversible = SequenceReversible(*reversible)
#         self.reversible = reversible

#     def step_x(self, x: IO, t: IO, state: State) -> IO:
#         """Update x

#         Args:
#             x (IO): Input
            
#         Returns:
#             IO: The updated input
#         """
#         return iou(self.reversible.reverse(t.f))

#     def step(self, x: IO, t: IO, state: State):
#         """These layers do not have parameters so the internal mechanics are not updated

#         Args:
#             x (IO): The input
#             t (IO): The output 
#         """
#         pass

#     def forward_nn(self, x: IO, state: State, **kwargs) -> typing.Union[typing.Tuple, typing.Any]:
#         return self.reversible(x.f)
